"""重排序客户端：百炼原生 rerank API（默认 qwen3-vl-rerank）或 none（RRF 直出）。

实测备注（2026-08-07）：百炼 rerank 服务需在控制台开通，sk-ws 密钥未开通时
返回 Access denied / url error；届时可切换 rerank_backend=none 降级，或开通后启用。
"""

from __future__ import annotations

import threading
from typing import Protocol

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class Reranker(Protocol):
    def rerank(self, query: str, documents: list[str]) -> list[float]: ...


class ApiReranker:
    """百炼原生 rerank：POST /api/v1/services/rerank/text-rerank/generation。"""

    name = "api"

    def __init__(self, base_url: str, api_key: str, model: str):
        if not api_key:
            raise RuntimeError("RERANK_API_KEY 未配置，无法使用 api 重排序后端")
        import httpx

        self._httpx = httpx
        self._base_url = base_url
        self._api_key = api_key
        self._model = model

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        try:
            resp = self._httpx.post(
                f"{self._base_url}/api/v1/services/rerank/text-rerank/generation",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "input": {"query": query, "documents": documents},
                    "parameters": {"top_n": len(documents)},
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            # rerank 失败不阻断问答：返回全零分，调用方保持 RRF 顺序
            logger.warning("rerank 调用失败，跳过精排（保持 RRF 顺序）: %s", e)
            return [0.0] * len(documents)
        results = data.get("output", {}).get("results", [])
        scores = [0.0] * len(documents)
        for item in results:
            idx = item.get("index")
            if idx is not None and 0 <= idx < len(documents):
                scores[idx] = float(item.get("relevance_score", 0.0))
        return scores


class NoneReranker:
    """占位：跳过重排序，RRF 结果直出。"""

    name = "none"

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        return [0.0] * len(documents)


_lock = threading.Lock()
_instance: Reranker | None = None


def get_reranker() -> Reranker:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                settings = get_settings()
                backend = settings.rerank_backend
                if backend == "api":
                    _instance = ApiReranker(
                        settings.rerank_api_base,
                        settings.rerank_api_key,
                        settings.rerank_model,
                    )
                elif backend == "none":
                    _instance = NoneReranker()
                else:
                    raise ValueError(f"未知的 rerank_backend: {backend}（可选：api | none）")
                logger.info("reranker ready: %s (%s)", _instance.name, settings.rerank_model)
    return _instance
