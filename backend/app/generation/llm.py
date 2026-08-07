"""LLM 客户端：OpenAI 兼容（DeepSeek/Qwen）+ 开发用 Mock。流式输出。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


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
    ) -> str:
        """一次性返回完整文本。response_format 支持 {"type": "json_object"} 等。"""
        ...


class OpenAICompatClient(LLMClient):
    name = "openai"

    def __init__(self, base_url: str, api_key: str, timeout: int, model: str):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self._default_model = model

    async def stream_chat(self, messages: list[dict[str, str]], model: str | None = None) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
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
    ) -> str:
        kwargs = {}
        if response_format:
            kwargs["response_format"] = response_format
        resp = await self._client.chat.completions.create(
            model=model or self._default_model,
            messages=messages,
            stream=False,
            **kwargs,
        )
        return resp.choices[0].message.content or ""


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
    ) -> str:
        return self._compose(messages)

    def _compose(self, messages: list[dict[str, str]]) -> str:
        # 取最后一条 user 消息中的知识块做简单模板回答
        last = messages[-1]["content"] if messages else ""
        import re

        m = re.search(r"【知识块】\n(.*?)(?:\n【|$)", last, re.S)
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
    if settings.llm_provider.lower() == "openai" and settings.llm_api_key:
        _instance = OpenAICompatClient(settings.llm_base_url, settings.llm_api_key, settings.llm_timeout, settings.llm_model)
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
    if not settings.llm_light_api_key:
        return None
    _light_instance = OpenAICompatClient(
        settings.llm_light_base_url, settings.llm_light_api_key, settings.llm_timeout, settings.llm_light_model
    )
    logger.info("LLM light client: openai (%s)", settings.llm_light_model)
    return _light_instance
