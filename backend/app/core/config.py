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

    # API 认证（HMAC 签名 + 时间戳窗口 + nonce 防重放；默认关闭，开启后 /api/v1/* 需携带签名头）
    auth_enabled: bool = False
    auth_client_key: str = ""  # 客户端标识（明文头）
    auth_secret: str = ""  # 共享密钥（仅服务端持有，验签用）
    auth_timestamp_window: int = 300  # 时间戳允许偏差（秒），超出视为过期/重放
    auth_nonce_ttl: int = 300  # nonce 去重保留时长（秒）

    # 存储
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "chunks"
    redis_url: str = "redis://localhost:6379"

    # 阶段四：异步任务后端（background=FastAPI BackgroundTasks 进程内线程池，
    # celery=独立 worker 进程，broker=redis_url；无 worker 环境/测试用 background）
    task_backend: str = "background"
    # 阶段四：OTel 可观测（trace/metrics 本地导出——ConsoleSpanExporter + ConsoleMetricReader）
    otel_enabled: bool = False

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
    # 会话/任务级超时（超时后发送 error 事件 / 标记任务失败）
    chat_timeout: int = 180  # SSE 问答整条流总超时（秒）
    ingest_timeout: int = 1800  # 单文档解析入库总超时（秒）

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
    rerank_candidates: int = 8  # RRF 融合后送入 rerank 的候选数（2026-08-11 消融：候选越少质量越高，8 最优，见 ablation §13）
    # 字段回填保底：relay 块（字段索引精确指路的答案块）rerank 分数加成，
    # 防止被语义打分挤出 top_k（字段确定性 > rerank 概率性，见 2026-08-10 消融）
    field_relay_boost: float = 0.35
    # 召回多样性：稠密路对近重复内容去重（同一表格/段落多块只保留最高分），提升候选覆盖
    retrieval_diversity_enabled: bool = True
    # 精确短语路由到稀疏路（§17）：引号/书名号短语独立嵌入稀疏检索、并入 RRF——
    # 对抗题"断言核实"里专有名词/制度名在整句稀疏向量中被稀释、相关块未召回，
    # 短语独立成路可锚定召回（验证后决定生产默认开关）
    phrase_route_enabled: bool = True
    phrase_route_top_k: int = 10  # 短语路单短语稀疏检索 top_k
    # 数字+单位短语精确索引硬插（离线构建 scripts/build_number_phrase_index.py）：
    # query 短语精确命中后硬插候选集（embedding 对精确数字不敏感），最多插入条数
    number_phrase_max_insert: int = 5
    # 章节父块检索（§18）：综述/总结类问题额外召回 chunk_type=section 父块并入 RRF——
    # 相关块分散多页时叶子单点命中率低，父块命中代表"章节整体相关"（叶子相关集
    # 评测口径扩展：父块计入相关集）。池重构后 top_k=12：综述题相关集主体是父块
    # （nParent 5~9 / nRel 6~19）。12 vs 10 的取舍：12+6 池下限更高（rerank 降级时
    # 父块在第 7-8 槽直接进 top-8，no-rerank top8 相关块 0.35→0.96；10+8 仅池上限
    # 微优 0.396 vs 0.385，且依赖完美排序假设）
    parent_route_top_k: int = 12
    # 综述题池重构（§18）：叶子 RRF 候选 8→6，槽位让给父块（父块12+叶子6=18 槽）。
    # 探针实证：叶 8→6 仅损失 1 题相关叶入池，换来父块结构性进 top-8 的降级鲁棒性
    summary_leaf_candidates: int = 6
    # 父块 rerank 加分（§18.3）：父块 RRF 单路分低、进不了 rerank 候选——父块路命中
    # 直接注入候选，rerank 后加该分保底（对齐 field_relay_boost 机制；实测候选外 100%）
    parent_route_boost: float = 0.25
    # 父块上下文治理：单块父块内容 token 上限（超长章节截锚点前后窗口；命中子块
    # 锚点 <hit> 标记 + 同父块只附一份全文，见 _attach_parent_context）
    parent_max_tokens: int = 2048
    # HyDE 稠密融合权重（原查询 0.7 / HyDE 0.3）：仅当场景分流允许时生效
    hyde_dense_weight: float = 0.3
    # 数值范围约束（六）：解析"指标+比较符+阈值"，候选块同族数值满足度作为奖励分
    numeric_range_enabled: bool = True
    numeric_range_boost: float = 0.2
    # 章节类型权重（七）：核心财务 1.2 / 业务分析 1.1 / 风险 1.0 / 目录释义 0.7（仅 rerank 路径）
    section_weighting_enabled: bool = True
    # 对比题实体均衡保活（八）：检测到 ≥2 核心实体时按实体分组各取 Top-M
    comparison_balance_enabled: bool = True

    # 召回不足分级降级兜底（十一）：top_k 内真实命中 < 阈值或候选池 < 下限时，
    # 按 L1短语硬插→L2年份±1→L2无年份→L3实体→L4纯dense 逐级放宽，达标即停。
    fallback_enabled: bool = True
    fallback_real_hit_threshold: int = 2   # top_k 内真实命中（非 relay/硬插/父块）阈值
    fallback_min_candidates: int = 6       # 返回条数下限（候选池过空辅助触发）
    fallback_year_window: int = 1          # L2 年份前后放宽窗口（±1 年）
    fallback_max_level: int = 4            # 最大降级级数（L1~L4）
    fallback_stall_stop: bool = True       # 连续无改善熔断（防越放宽越偏）

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
    clean_page_numbers: bool = True  # 页码行剔除（纯数字 / 第X页 / Page X，仅限页首/页尾边缘区）
    upload_dir: str = "./data/uploads"
    pipeline_dir: str = "./data/pipeline"
    registry_db: str = "./data/registry.db"

    # 会话（阶段二启用）
    session_window: int = 10
    semantic_cache_threshold: float = 0.95  # 语义答案缓存相似度阈值（复用）
    # 语义答案缓存二次校验：向量命中后，问题文本字符重合度低于该值视为可疑（防御误命中）
    answer_cache_min_overlap: float = 0.5
    # 问答记录持久化（chat_records 表，供离线复评 / RAGAS）
    qa_record_enabled: bool = True

    # Agent 防线（Guard 层，轻量版）
    guard_enabled: bool = True  # 开启注入/循环检测
    guard_injection_trigger_words: str = ""  # 额外注入关键词（逗号分隔，追加到正则特征）
    guard_loop_max_repeats: int = 3  # 同问题归一化重复达该次数判定循环

    # 外部 API 缓解（缓存 + 限流 + 重试）
    semantic_cache_enabled: bool = True  # 语义答案缓存（仅无 session_id 的单轮问答生效）
    embed_cache_ttl: int = 86400  # 嵌入结果缓存秒数（相同文本复用向量，省外部调用）
    answer_cache_ttl: int = 3600  # 答案缓存秒数
    answer_cache_max_entries: int = 50  # 每 org+visibility 桶内最大缓存条目
    max_embed_concurrency: int = 8  # 嵌入 API 并发上限（超限排队，防 429；实测 8 为吞吐/稳定性平衡点）
    max_llm_concurrency: int = 4  # LLM 并发上限
    max_rerank_concurrency: int = 4  # rerank 并发上限

    # 文档侧 IDF 重写（稀疏路治本）：入库时对稀疏 index 乘 IDF，压低高频结构词、抬升判别词
    sparse_idf_enabled: bool = True
    sparse_idf_path: str = "./data/idf.json"  # 离线统计的 index→IDF 映射

    # 结构块（财务表头/页眉/版式说明）：检索时置后降权（占比极小，实测对召回无影响，保留单一行为）
    structural_chunk_enabled: bool = True

    # 字段抽取（结构化指标 → 独立索引）：入库时抽取财务指标，支撑"某年某指标"类问答精确取值
    fields_enabled: bool = True

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

    @property
    def resolved_sparse_idf_path(self) -> Path:
        p = Path(self.sparse_idf_path)
        return p if p.is_absolute() else ROOT_DIR / p


@lru_cache
def get_settings() -> Settings:
    return Settings()
