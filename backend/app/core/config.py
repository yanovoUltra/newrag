"""应用配置：全部通过 .env / 环境变量管理。"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> backend/ -> 项目根目录
BACKEND_DIR = Path(__file__).resolve().parents[2]
ROOT_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 服务
    app_env: str = "development"
    app_port: int = 8000
    cors_origins: str = "http://localhost:5173"

    # 存储
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "chunks"
    redis_url: str = "redis://localhost:6379"

    # 嵌入（百炼 API 默认 / bge-m3 本地）
    embedding_model: str = "qwen3.7-text-embedding"
    embedding_backend: str = "api"  # api | flagembedding | mock
    embedding_device: str = "cpu"
    embedding_cache_dir: str = "./data/models"
    # API 后端（OpenAI 兼容，如百炼 /compatible-mode/v1）
    embedding_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_api_key: str = ""
    embedding_batch_size: int = 16

    # LLM
    llm_provider: str = "openai"  # openai | mock
    llm_base_url: str = "https://api.deepseek.com"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_timeout: int = 60
    # 多模型路由：简单问题走轻量模型（如百炼 qwen3.5-flash），未配置则全部走主模型
    llm_light_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_light_api_key: str = ""
    llm_light_model: str = "qwen3.5-flash"

    # OCR（百度智能云 / PaddleOCR AI Studio）
    ocr_enabled: bool = False
    ocr_provider: str = "baidu"  # baidu | ppocr
    ocr_api_key: str = ""
    ocr_secret_key: str = ""
    # PaddleOCR AI Studio 在线 API（PP-StructureV3，异步任务：提交 → 轮询 → 下载 JSONL）
    ppocr_token: str = ""
    ppocr_job_url: str = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
    ppocr_model: str = "PP-StructureV3"
    ppocr_poll_interval: int = 5  # 轮询间隔（秒）
    ppocr_timeout: int = 60  # 单次 HTTP 超时（秒）

    # 检索（阶段二：混合检索 + 重排序）
    recall_dense_top_k: int = 50
    recall_sparse_top_k: int = 50
    recall_table_top_k: int = 20
    rrf_k: int = 60
    retrieval_top_k: int = 8
    # 重排序：api（百炼 rerank，模型名可配） | none（RRF 直出）
    rerank_backend: str = "none"
    rerank_api_base: str = "https://dashscope.aliyuncs.com"
    rerank_api_key: str = ""
    rerank_model: str = "qwen3-rerank"
    rerank_api_path: str = "/api/v1/services/rerank/text-rerank/text-rerank"
    rerank_candidates: int = 50  # RRF 融合后送入 rerank 的候选数

    # 路由 / 查询重写 / HyDE
    intent_routing_enabled: bool = True
    query_rewrite_enabled: bool = True
    hyde_enabled: bool = True
    hyde_max_tokens: int = 256

    # 切分
    chunk_min_tokens: int = 128
    chunk_target_tokens: int = 200
    chunk_max_tokens: int = 256
    overlap_tokens: int = 20
    table_max_tokens: int = 2048  # 超过则表格"摘要+分页"

    # 上传
    max_upload_mb: int = 50
    upload_dir: str = "./data/uploads"
    pipeline_dir: str = "./data/pipeline"
    registry_db: str = "./data/registry.db"

    # 会话（阶段二启用）
    session_window: int = 10
    semantic_cache_threshold: float = 0.95

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def resolved_upload_dir(self) -> Path:
        p = Path(self.upload_dir)
        return p if p.is_absolute() else ROOT_DIR / p

    @property
    def resolved_pipeline_dir(self) -> Path:
        p = Path(self.pipeline_dir)
        return p if p.is_absolute() else ROOT_DIR / p

    @property
    def resolved_registry_db(self) -> Path:
        p = Path(self.registry_db)
        return p if p.is_absolute() else ROOT_DIR / p

    @property
    def resolved_embedding_cache_dir(self) -> Path:
        p = Path(self.embedding_cache_dir)
        return p if p.is_absolute() else ROOT_DIR / p


@lru_cache
def get_settings() -> Settings:
    return Settings()
