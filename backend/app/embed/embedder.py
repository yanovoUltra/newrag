"""嵌入层：稠密向量。后端可切换（百炼 API 默认 / flagembedding 本地 / mock 测试）。"""

from __future__ import annotations

import hashlib
import threading
from typing import Protocol

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

EMBED_DIM = 1024  # bge-m3 / qwen3.7-text-embedding 稠密维度


class EmbeddingBackend(Protocol):
    name: str

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量稠密向量，L2 归一化。"""
        ...


class MockEmbeddingBackend:
    """确定性伪向量：用于测试与无模型环境，不具备真实语义。"""

    name = "mock"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vecs = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            v = np.zeros(EMBED_DIM, dtype=np.float32)
            for i, b in enumerate(h):
                v[i % EMBED_DIM] += b
            norm = np.linalg.norm(v) or 1.0
            vecs.append((v / norm).tolist())
        return vecs


class FlagEmbeddingBackend:
    """FlagEmbedding（torch）：BAAI/bge-m3，稠密+稀疏同模型。"""

    name = "flagembedding"

    def __init__(self, model_name: str, cache_dir, device: str):
        from FlagEmbedding import BGEM3FlagModel

        self._model = BGEM3FlagModel(
            model_name,
            use_fp16=False,
            device=device,
            cache_dir=str(cache_dir),
        )
        self._model_name = model_name

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out = self._model.encode(texts, return_dense=True, return_sparse=False)["dense_vecs"]
        return np.asarray(out, dtype=np.float32).tolist()


class ApiEmbedBackend:
    """百炼 OpenAI 兼容 API：qwen3.7-text-embedding，1024 维，无需本地权重。"""

    name = "api"

    def __init__(self, base_url: str, api_key: str, model: str, batch_size: int):
        if not api_key:
            raise RuntimeError("EMBEDDING_API_KEY 未配置，无法使用 api 嵌入后端（请在 .env 中配置）")
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=60)
        self._model = model
        self._batch_size = batch_size

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vecs: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            resp = self._client.embeddings.create(
                model=self._model,
                input=batch,
                encoding_format="float",
            )
            for item in sorted(resp.data, key=lambda d: d.index):
                emb = item.embedding
                if len(emb) != EMBED_DIM:
                    raise RuntimeError(
                        f"embedding 维度不匹配：{self._model} -> {len(emb)}，期望 {EMBED_DIM}，"
                        "需同步调整 Qdrant collection 配置"
                    )
                vecs.append(emb)
        return vecs


_lock = threading.Lock()
_instance: EmbeddingBackend | None = None


def get_embedder() -> EmbeddingBackend:
    """懒加载单例；模型权重首次使用时下载。"""
    global _instance
    if _instance is not None:
        return _instance
    with _lock:
        if _instance is not None:
            return _instance
        settings = get_settings()
        cache_dir = settings.resolved_embedding_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        backend = settings.embedding_backend.lower()
        if backend == "mock":
            _instance = MockEmbeddingBackend()
        elif backend == "api":
            _instance = ApiEmbedBackend(
                settings.embedding_api_base,
                settings.embedding_api_key,
                settings.embedding_model,
                settings.embedding_batch_size,
            )
        elif backend == "flagembedding":
            _instance = FlagEmbeddingBackend(settings.embedding_model, cache_dir, settings.embedding_device)
        else:
            raise ValueError(f"未知的 embedding_backend: {backend}（可选：api | flagembedding | mock）")
        logger.info("embedding backend ready: %s (%s)", _instance.name, settings.embedding_model)
        return _instance
