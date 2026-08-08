"""嵌入层：稠密 + 稀疏向量。后端可切换（百炼 API 默认 / flagembedding 本地 / mock 测试）。

API 后端使用 DashScope 原生接口（/api/v1/services/embeddings/...）：
- OpenAI 兼容端点不支持稀疏向量，原生接口通过 output_type=dense&sparse 一次返回稠密+稀疏；
- 稀疏向量为模型词表 token 级权重 [{index, value, token}]，index 可直接作为 Qdrant 稀疏索引；
- text_type 区分 document（入库）与 query（检索），提升检索质量。
"""

from __future__ import annotations

import hashlib
import threading
from typing import Protocol

import httpx
import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

EMBED_DIM = 1024  # qwen3.7-text-embedding / text-embedding-v4 默认稠密维度


class EmbeddingBackend(Protocol):
    name: str

    def embed_texts(self, texts: list[str], *, text_type: str = "document") -> list[list[float]]:
        """批量稠密向量（L2 归一化由模型保证）。"""
        ...

    def embed_texts_with_sparse(
        self, texts: list[str], *, text_type: str = "document"
    ) -> tuple[list[list[float]], list[dict] | None]:
        """批量稠密 + 稀疏向量；后端不支持稀疏时稀疏部分返回 None。"""
        ...


class MockEmbeddingBackend:
    """确定性伪向量：用于测试与无模型环境，不具备真实语义。"""

    name = "mock"

    def embed_texts(self, texts: list[str], *, text_type: str = "document") -> list[list[float]]:
        vecs = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            v = np.zeros(EMBED_DIM, dtype=np.float32)
            for i, b in enumerate(h):
                v[i % EMBED_DIM] += b
            norm = np.linalg.norm(v) or 1.0
            vecs.append((v / norm).tolist())
        return vecs

    def embed_texts_with_sparse(
        self, texts: list[str], *, text_type: str = "document"
    ) -> tuple[list[list[float]], None]:
        return self.embed_texts(texts, text_type=text_type), None


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

    def embed_texts(self, texts: list[str], *, text_type: str = "document") -> list[list[float]]:
        out = self._model.encode(texts, return_dense=True, return_sparse=False)["dense_vecs"]
        return np.asarray(out, dtype=np.float32).tolist()

    def embed_texts_with_sparse(
        self, texts: list[str], *, text_type: str = "document"
    ) -> tuple[list[list[float]], None]:
        # bge-m3 稀疏权重可在此扩展；当前统一由 API 后端提供原生稀疏
        return self.embed_texts(texts, text_type=text_type), None


def _native_base(base_url: str) -> str:
    """OpenAI 兼容端点（.../compatible-mode/v1）→ DashScope 原生端点（.../api/v1）。"""
    if base_url.endswith("/compatible-mode/v1"):
        return base_url.replace("/compatible-mode/v1", "/api/v1")
    return base_url


def _parse_native_response(data: dict, model: str) -> tuple[list[list[float]], list[dict] | None]:
    """解析 DashScope 原生 /text-embedding 响应 → (稠密列表, 稀疏列表|None)。

    响应结构：output.embeddings[] = {embedding: [...], sparse_embedding: [{index,value,token}], text_index}
    稀疏字段缺失（模型/端点未开启）时返回 None。
    """
    embeddings = data.get("output", {}).get("embeddings", [])
    ordered = sorted(embeddings, key=lambda e: e.get("text_index", 0))
    dense: list[list[float]] = []
    sparse: list[dict] | None = []
    for item in ordered:
        emb = item.get("embedding")
        if emb is None:
            raise RuntimeError(f"embedding 响应缺少稠密向量（model={model}）")
        dense.append(emb)
        sp = item.get("sparse_embedding") or item.get("sparse_embeddings")
        if sp:
            sparse.append(
                {
                    "indices": [int(s["index"]) for s in sp],
                    "values": [float(s["value"]) for s in sp],
                }
            )
        else:
            sparse = None
    if sparse is None or not sparse:
        sparse = None
    return dense, sparse


class ApiEmbedBackend:
    """百炼 DashScope 原生 API：qwen3.7-text-embedding，稠密 1024 + 模型原生稀疏。"""

    name = "api"

    def __init__(self, base_url: str, api_key: str, model: str, batch_size: int):
        if not api_key:
            raise RuntimeError("EMBEDDING_API_KEY 未配置，无法使用 api 嵌入后端（请在 .env 中配置）")
        self._base = _native_base(base_url)
        self._key = api_key
        self._model = model
        self._batch_size = batch_size
        self._client = httpx.Client(timeout=60)

    def _call(self, texts: list[str], output_type: str, text_type: str) -> tuple[list[list[float]], list[dict] | None]:
        # DashScope 原生 API 单次 input.texts 上限 20，按配置 batch_size 分批调用后拼接
        dense_all: list[list[float]] = []
        sparse_all: list[dict] | None = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            resp = self._client.post(
                f"{self._base}/services/embeddings/text-embedding/text-embedding",
                headers={"Authorization": f"Bearer {self._key}"},
                json={
                    "model": self._model,
                    "input": {"texts": batch},
                    "parameters": {"output_type": output_type, "dimension": EMBED_DIM, "text_type": text_type},
                },
            )
            if resp.status_code != 200:
                raise RuntimeError(f"embedding API {resp.status_code}: {resp.text[:300]}")
            dense, sparse = _parse_native_response(resp.json(), self._model)
            for emb in dense:
                if len(emb) != EMBED_DIM:
                    raise RuntimeError(
                        f"embedding 维度不匹配：{self._model} -> {len(emb)}，期望 {EMBED_DIM}，"
                        "需同步调整 Qdrant collection 配置"
                    )
            dense_all.extend(dense)
            if sparse is not None:
                if sparse_all is not None:
                    sparse_all.extend(sparse)
            else:
                sparse_all = None
        return dense_all, (sparse_all or None)

    def embed_texts(self, texts: list[str], *, text_type: str = "document") -> list[list[float]]:
        dense, _ = self._call(texts, output_type="dense", text_type=text_type)
        return dense

    def embed_texts_with_sparse(
        self, texts: list[str], *, text_type: str = "document"
    ) -> tuple[list[list[float]], list[dict] | None]:
        # DashScope 原生接口 output_type=dense&sparse：一次返回稠密 + 模型原生稀疏
        return self._call(texts, output_type="dense&sparse", text_type=text_type)


_lock = threading.Lock()
_instance: EmbeddingBackend | None = None


def get_embedder() -> EmbeddingBackend:
    """懒加载单例；模型权重首次使用时下载（本地后端）。"""
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
