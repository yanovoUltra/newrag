"""LLM 客户端：OpenAI 兼容（DeepSeek/Qwen）+ 开发用 Mock。流式输出。"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.metrics import record_api_call
from app.core.secrets import resolve_secret

logger = get_logger(__name__)


def _thinking_extra_body(base_url: str, enabled: bool) -> dict[str, Any]:
    """Return the provider-specific OpenAI-compatible thinking toggle."""

    host = base_url.casefold()
    if "deepseek.com" in host:
        return {"thinking": {"type": "enabled" if enabled else "disabled"}}
    if "aliyuncs.com" in host:
        return {"enable_thinking": enabled}
    return {}


class FairSemaphore:
    """FIFO 公平信号量：按到达顺序放行。

    asyncio.Semaphore 非公平——当大量协程同时涌入且配额不足时，后到者可能
    先拿到配额，先到者被饿在队尾（实测并发 6 问时个别请求被拖到 5 倍延迟）。
    检索段线程池化后，多个请求会同时到达 LLM 段，因此需要严格先来先服务。
    """

    def __init__(self, limit: int):
        self._limit = max(limit, 1)
        self._active = 0
        self._queue: list[asyncio.Future] = []

    async def acquire(self) -> None:
        if self._active < self._limit:
            self._active += 1
            return
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._queue.append(fut)
        try:
            await fut
        except asyncio.CancelledError:
            if fut in self._queue:
                self._queue.remove(fut)
            raise

    def release(self) -> None:
        if self._active > 0:
            self._active -= 1
        while self._queue and self._active < self._limit:
            fut = self._queue.pop(0)
            self._active += 1
            if not fut.done():
                fut.set_result(None)

    async def __aenter__(self) -> "FairSemaphore":
        await self.acquire()
        return self

    async def __aexit__(self, *exc) -> None:
        self.release()


class LLMClient(ABC):
    name: str

    @abstractmethod
    async def stream_chat(self, messages: list[dict[str, str]], model: str | None = None) -> AsyncIterator[str]:
        """流式返回增量文本。"""
        ...

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        response_format: dict | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        thinking: bool | None = None,
    ) -> str:
        """一次性返回完整文本。response_format 支持 {"type": "json_object"} 等。"""
        ...

    async def tool_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str | None = None,
    ) -> dict[str, Any]:
        """Return normalized content/tool calls; providers may opt out."""

        raise NotImplementedError("This LLM backend does not support tool calling")


class OpenAICompatClient(LLMClient):
    name = "openai"

    def __init__(self, base_url: str, api_key: str, timeout: int, model: str):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self._base_url = base_url
        self._default_model = model
        # 外部 API 缓解：并发限流（FIFO 公平排队，防 429 与信号量饥饿）+ 429/5xx 指数退避重试
        self._sem = FairSemaphore(get_settings().max_llm_concurrency)

    async def _create_with_retry(self, **kwargs: Any):
        """限流 + 重试的 create 调用；重试 3 次后抛出最后一次异常。"""
        async with self._sem:
            t0 = time.perf_counter()
            for attempt in range(3):
                try:
                    resp = await self._client.chat.completions.create(**kwargs)
                    record_api_call("llm", time.perf_counter() - t0)
                    return resp
                except Exception as e:
                    status = getattr(e, "status_code", None)
                    if status in (429, 500, 502, 503, 504) and attempt < 2:
                        await asyncio.sleep(0.5 * (attempt + 1))
                        continue
                    record_api_call("llm", time.perf_counter() - t0, error=True)
                    raise

    async def stream_chat(self, messages: list[dict[str, str]], model: str | None = None) -> AsyncIterator[str]:
        stream = await self._create_with_retry(
            model=model or self._default_model,
            messages=messages,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        response_format: dict | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        thinking: bool | None = None,
    ) -> str:
        kwargs = {}
        if response_format:
            kwargs["response_format"] = response_format
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        if thinking is not None:
            extra_body = _thinking_extra_body(self._base_url, thinking)
            if extra_body:
                kwargs["extra_body"] = extra_body
        resp = await self._create_with_retry(
            model=model or self._default_model,
            messages=messages,
            stream=False,
            **kwargs,
        )
        return resp.choices[0].message.content or ""

    async def tool_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str | None = None,
    ) -> dict[str, Any]:
        extra_body = _thinking_extra_body(self._base_url, False)
        resp = await self._create_with_retry(
            model=model or self._default_model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            stream=False,
            **({"extra_body": extra_body} if extra_body else {}),
        )
        message = resp.choices[0].message
        calls = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in (message.tool_calls or [])
        ]
        return {"content": message.content or "", "tool_calls": calls}


def _status_code(exc: Exception) -> int | None:
    value = getattr(exc, "status_code", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


class ForbiddenFallbackClient(LLMClient):
    """Use the secondary Token Plan model only when the primary is forbidden.

    A 403 typically means the selected package does not grant access to the
    requested model. Other errors retain the normal retry/failure semantics so
    quota, timeout, and server faults are not silently hidden by model routing.
    """

    name = "openai-forbidden-fallback"

    def __init__(self, primary: LLMClient, fallback: LLMClient):
        self._primary = primary
        self._fallback = fallback

    async def stream_chat(
        self, messages: list[dict[str, str]], model: str | None = None
    ) -> AsyncIterator[str]:
        emitted = False
        try:
            async for chunk in self._primary.stream_chat(messages, model=model):
                emitted = True
                yield chunk
        except Exception as exc:
            if emitted or _status_code(exc) != 403:
                raise
            logger.warning("primary LLM returned 403; switching to configured fallback model")
            async for chunk in self._fallback.stream_chat(messages):
                yield chunk

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        response_format: dict | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        thinking: bool | None = None,
    ) -> str:
        kwargs = {
            "response_format": response_format,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "thinking": thinking,
        }
        try:
            return await self._primary.chat(messages, model=model, **kwargs)
        except Exception as exc:
            if _status_code(exc) != 403:
                raise
            logger.warning("primary LLM returned 403; switching to configured fallback model")
            return await self._fallback.chat(messages, **kwargs)

    async def tool_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str | None = None,
    ) -> dict[str, Any]:
        try:
            return await self._primary.tool_chat(messages, tools, model=model)
        except Exception as exc:
            if _status_code(exc) != 403:
                raise
            logger.warning("primary LLM returned 403; switching to configured fallback model")
            return await self._fallback.tool_chat(messages, tools)


class MockLLMClient(LLMClient):
    """开发用 Mock：无 API 密钥时可用，仅基于检索上下文生成模板式回答，禁止生产使用。"""

    name = "mock"

    async def stream_chat(self, messages: list[dict[str, str]], model: str | None = None) -> AsyncIterator[str]:
        answer = self._compose(messages)
        for token in answer:
            yield token

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        response_format: dict | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        thinking: bool | None = None,
    ) -> str:
        return self._compose(messages)

    async def tool_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str | None = None,
    ) -> dict[str, Any]:
        return {"content": self._compose(messages), "tool_calls": []}

    def _compose(self, messages: list[dict[str, str]]) -> str:
        # 取最后一条 user 消息中的知识块做简单模板回答
        last = messages[-1]["content"] if messages else ""
        import re

        m = re.search(r"【知识块\s*\d*】\n(.*?)(?:\n【|$)", last, re.S)
        if not m:
            return "（Mock LLM）未配置 API 密钥，无法生成基于真实模型的回答。"
        snippet = m.group(1).strip().replace("\n", " ")[:200]
        return (
            "（Mock LLM 回答，开发验证用）根据提供的财报知识块，相关信息如下：\n"
            f"{snippet}\n"
            "提示：配置 LLM_API_KEY 后获得真实流式回答。"
        )


_instance: LLMClient | None = None
_light_instance: LLMClient | None = None


def get_llm() -> LLMClient:
    global _instance
    if _instance is not None:
        return _instance
    settings = get_settings()
    api_key = resolve_secret(settings.llm_api_secret_name, settings.llm_api_key)
    if settings.llm_provider.lower() == "openai" and api_key:
        primary = OpenAICompatClient(
            settings.llm_base_url,
            api_key,
            settings.llm_timeout,
            settings.llm_model,
        )
        fallback_key = resolve_secret(
            settings.llm_light_api_secret_name, settings.llm_light_api_key
        )
        if fallback_key and settings.llm_light_model:
            fallback = OpenAICompatClient(
                settings.llm_light_base_url,
                fallback_key,
                settings.llm_timeout,
                settings.llm_light_model,
            )
            _instance = ForbiddenFallbackClient(primary, fallback)
            logger.info(
                "LLM client: openai (%s; 403 fallback=%s)",
                settings.llm_model,
                settings.llm_light_model,
            )
        else:
            _instance = primary
            logger.info("LLM client: openai (%s)", settings.llm_model)
    else:
        logger.warning("LLM_API_KEY 未配置，使用 MockLLMClient（仅开发验证）")
        _instance = MockLLMClient()
    return _instance


def get_llm_light() -> LLMClient | None:
    """轻量模型客户端（多模型路由）：未配置 LLM_LIGHT_API_KEY 时返回 None（全部走主模型）。"""
    global _light_instance
    if _light_instance is not None:
        return _light_instance
    settings = get_settings()
    api_key = resolve_secret(
        settings.llm_light_api_secret_name, settings.llm_light_api_key
    )
    if not api_key:
        return None
    _light_instance = OpenAICompatClient(
        settings.llm_light_base_url,
        api_key,
        settings.llm_timeout,
        settings.llm_light_model,
    )
    logger.info("LLM light client: openai (%s)", settings.llm_light_model)
    return _light_instance
