# NewRAG 项目经历（简历版）

> 口径说明：以下只采用当前源码、评测报告和已完成部署可验证的事实。`Recall@500 > 0.9` 不写成 `NDCG@8 > 0.9`；公网生产当前为 Mock 模型演示配置，不表述为真实模型在线服务。

## ★ NewRAG — 多模态财报智能问答与检索系统

**项目状态：** 已完成本地全栈、容器化生产部署与可信 HTTPS ｜ **角色：** AI 应用开发 / RAG 检索优化 / 全栈工程化  
**项目定位：** 面向中英文财报的多格式 RAG 平台，支持 PDF、DOCX、XLSX、图片解析入库，提供结构化指标检索、多证据问答、流式输出和来源引用。  
**技术栈：** 后端： Python 3 + FastAPI + Celery + SQLAlchemy + OpenAI 兼容 API / 存储： SQLite + Qdrant + Redis / AI 检索增强： qwen3-rerank + RRF + HyDE / 前端： Vue 3 + TypeScript + Vite / 部署： Docker Compose + Nginx + Caddy + Certbot / 可观测： OpenTelemetry
### 我的工作

- ● 负责财报 RAG 全链路设计与实现，搭建“多格式解析 → OCR 按需触发 → NFKC/页眉页脚清洗 → 章节树 → Small-to-Big 分块 → dense+sparse 向量化 → Qdrant 入库 → SSE 问答”的完整流水线；支持阶段落盘、断点续跑、失败向量回滚和 Celery 异步入库。
- ● 设计 dense / sparse / table / 财务字段 relay / 数字短语 / 章节父块多路候选与 RRF 融合，加入公司、年份、权限、否定词和数值范围约束；将非指标题大候选池 Recall 提升至 **0.9232**，父章节 Recall@100 达 **0.9968**，并通过消融确认瓶颈由召回转移到多方面排序。
- ● 构建 `financial_fields` 独立索引，以“公司 + 指标 + 年份”确定性约束回填权威来源块，避免概率精排误排；168 道指标题 **NDCG@1 / Precision@1 / MRR 均为 1.0000**，指标科目、年份、数值抽取准确率分别为 **98.21% / 100% / 98.21%**。
- ● 搭建 239 题中英文检索评测与 holdout 泛化体系，覆盖 NDCG、Recall、Precision、MRR、严格 All-Hit、可达子集 All-Hit 和 Parent All-Hit；以内容 SHA-256 指纹解耦易变 chunk_id，并用消融实验拒绝 BM25、MMR、盲目扩大候选等负优化方案。
- ● 优化入库与运行性能：修复 embedding 批调用串行问题，通过并发保序将历史吞吐由 **3.1 提升至 9.9 texts/s（约 3.2 倍）**；上传改为 1MB 分块落盘并同步 SHA-256，向量化/Qdrant 写入按 128 块分批，文档 embedding 默认不写 Redis，缓存与 Celery broker 拆分并采用不同淘汰策略。
- ● 优化前端流式体验与可访问性：SSE token 按帧合并渲染，滚动和 localStorage 写入节流，历史限制为最近 20 轮或 1MB；实现服务端分页/筛选与过期请求取消、移动端卡片/底部导航、44px 点击区、语义化 label、状态播报和 axe 三视口回归。
- ● 完成生产工程化与公网部署：构建 API、Celery worker、Vue/Nginx、Qdrant、缓存 Redis、broker Redis、Caddy **7 服务** Compose；仅暴露 80/443，配置 Basic Auth、安全响应头、日志轮转、资源限制、持久卷、健康检查及 Let's Encrypt 公网 IP 可信 HTTPS 自动续期。
- ● 建立质量与可观测基线：请求 trace id、JSON 日志、可配置 OTLP/采样、livez 与 5 秒缓存 healthz；完成后端 **167 项**、前端 **13 项**、Playwright+axe **9 个页面/视口组合**及生产构建验收，主入口 JS gzip 约 **148.1KB**。

### 量化成果框

| 维度 | 结果 |
| --- | ---: |
| 指标题最终排序 | NDCG@1 / P@1 / MRR = **1.0000 / 1.0000 / 1.0000**（168 题） |
| 非指标深召回 | dense@500 ∪ sparse@500 Recall = **0.9232** |
| 分层章节召回 | Parent Recall@50 / @100 = **0.9367 / 0.9968** |
| 非指标稳定生产路径 | NDCG@8 / Recall@8 / MRR = **0.5045 / 0.4807 / 0.8404** |
| 入库吞吐优化 | **3.1 → 9.9 texts/s，约 3.2×** |
| 工程验证 | 后端 **167** + 前端 **13** + axe/E2E **9**，生产构建通过 |

## 一页简历压缩版（建议保留 4 条）

**★ NewRAG — 多模态财报 RAG 系统｜AI 应用开发工程师**  
`FastAPI / Celery / Qdrant / Redis / SQLAlchemy / RRF / qwen3-rerank / Vue 3 / Docker Compose`

- ● 设计 dense/sparse/table/字段 relay/数字短语/章节父块多路检索与 RRF 融合，将非指标大候选池 Recall 提升至 **0.9232**、Parent Recall@100 提升至 **0.9968**，通过 239 题消融将瓶颈定位到多方面精排。
- ● 构建公司+指标+年份结构化字段索引并确定性回填权威证据，168 道财报指标题 **NDCG@1、P@1、MRR 均达 1.0**，指标/年份/数值抽取准确率 **98.21%/100%/98.21%**。
- ● 完成解析、OCR、Small-to-Big 切块、批量向量化、Qdrant 入库、SSE 生成与 grounding 全链路；修复 embedding 并发，将吞吐由 **3.1 提升至 9.9 texts/s（3.2×）**。
- ● 完成 7 服务生产 Compose 与公网可信 HTTPS：API/worker/frontend/Qdrant/双 Redis/Caddy 内网隔离，仅开放 80/443，并配置健康检查、资源限制、持久化、日志轮转和证书自动续期；后端 167 项、前端 13 项测试通过。

## 面试时应主动说明的真实边界

- 非指标题的 **0.9232 是大候选池 Recall，不是最终 Top-8 NDCG**；当前稳定 Top-8 为 NDCG 0.5045、Recall 0.4807、MRR 0.8404。
- 生产公网部署当前使用 Mock Embedding / Mock LLM / 无 rerank，证明的是完整部署、安全和运行链路；切换真实模型需要在服务端配置密钥并重新执行质量与成本验收。
- 当前 SQLite 适合单机单 API；面向高可用和水平扩容时，应迁移 PostgreSQL、对象存储和托管/集群化 Qdrant、Redis。

## 详细技术依据

完整教学与逐文件说明见同目录 `NewRAG_全链路技术教学报告.html`；检索指标的实验过程与口径见 `简历_检索召回量化报告.md`。
