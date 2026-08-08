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
    # API 后端（DashScope 原生 /api/v1，支持稠密+稀疏；OpenAI 兼容端点不返回稀疏向量）
    embedding_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_api_key: str = ""
    embedding_batch_size: int = 16

    # LLM
    llm_provider: str = "openai"  # openai | mock
    llm_base_url: str = "https://api.deepseek.com"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_timeout: int = 60
    # 多模型路由：简单问题走轻量模型（如阿里云套餐 qwen3.7-plus），未配置则全部走主模型
    llm_light_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_light_api_key: str = ""
    llm_light_model: str = "qwen3.7-plus"

    # OCR（PaddleOCR AI Studio 在线 API）
    ocr_enabled: bool = False
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
    # 置信度门控（仅 rerank 模式生效；RRF-only 模式请置 0）：top 命中分低于该值视为低置信
    retrieval_min_score: float = 0.3
    # 复杂/抽象/多跳问题检索加深系数（多路 top_k 与 rerank 候选数按该系数放大）
    retrieval_depth_scale: float = 1.5
    # 重排序：api（百炼 rerank，模型名可配） | none（RRF 直出）
    rerank_backend: str = "none"
    rerank_api_base: str = "https://dashscope.aliyuncs.com"
    rerank_api_key: str = ""
    rerank_model: str = "qwen3-rerank"
    rerank_api_path: str = "/api/v1/services/rerank/text-rerank/text-rerank"
    rerank_candidates: int = 50  # RRF 融合后送入 rerank 的候选数

    # 路由 / 查询重写 / HyDE
    intent_routing_enabled: bool = True
    hyde_enabled: bool = True

    # 切分
    chunk_min_tokens: int = 128
    chunk_target_tokens: int = 200
    chunk_max_tokens: int = 256
    chunk_parent_target_tokens: int = 3072  # 章节级父块目标（2~4K 区间中值）
    table_max_tokens: int = 2048  # 超过则表格"摘要+分页"
    # 语义精切（结构优先、语义为辅）：对超长散文叶子按相邻句相似度断点切分
    semantic_chunk_refine: bool = True
    semantic_split_sim_threshold: float = 0.5  # 相邻句相似度低于该值视为主题切换
    semantic_min_sentences: int = 6  # 句子数少于该值的叶子不精切

    # 上传 / 陈旧文档防护
    max_upload_mb: int = 50
    upload_replace_same_filename: bool = True  # 同名文件上传 → 替换旧版（先删旧点再入库）
    # 数据清洗（解析/OCR 之后、章节树之前）
    clean_enable: bool = True
    clean_nfkc: bool = True  # NFKC 归一化（全角→半角，统一数字/百分号/逗号）
    clean_header_footer: bool = True  # 页眉/页脚去重
    clean_header_footer_min_ratio: float = 0.5  # 行出现页数占比 >= 该值视为高频
    clean_header_footer_max_tokens: int = 24  # 高频行 token 上限（页眉/页脚通常为短行）
    clean_header_footer_margin: float = 0.15  # 页首/页尾边缘区比例（超出该区的行不删）
    upload_dir: str = "./data/uploads"
    pipeline_dir: str = "./data/pipeline"
    registry_db: str = "./data/registry.db"

    # 会话（阶段二启用）
    session_window: int = 10
    semantic_cache_threshold: float = 0.95  # 语义答案缓存相似度阈值（复用）
    # 语义答案缓存二次校验：向量命中后，问题文本字符重合度低于该值视为可疑（防御误命中）
    answer_cache_min_overlap: float = 0.5

    # 外部 API 缓解（缓存 + 限流 + 重试）
    semantic_cache_enabled: bool = True  # 语义答案缓存（仅无 session_id 的单轮问答生效）
    embed_cache_ttl: int = 86400  # 嵌入结果缓存秒数（相同文本复用向量，省外部调用）
    answer_cache_ttl: int = 3600  # 答案缓存秒数
    answer_cache_max_entries: int = 50  # 每 org+visibility 桶内最大缓存条目
    max_embed_concurrency: int = 4  # 嵌入 API 并发上限（超限排队，防 429）
    max_llm_concurrency: int = 4  # LLM 并发上限
    max_rerank_concurrency: int = 4  # rerank 并发上限

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
