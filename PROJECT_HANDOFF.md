# NewRAG：新会话交接与当前系统状态

核验日期：2026-09-09（北京时间）。先读本文；本地 `PROJECT_SUMMARY.md` 的历史条目不能覆盖本文的当前停点。

## 1. 当前目标与本轮边界

产品主线是 **找到证据 → 定位引用 → 正确回答**。提高真实 RAG 质量，争取目标指标 0.95，但不承诺所有指标已达到，也不能通过改评测合同制造达标。

- 指标题：沿用 **168 题**的检索/字段指标和回答质量结果；不要自动切到四公司 80 题。
- 非指标题：GroupRecall、All-Hit 优先，单独核对对应题集、分母和评价版本。
- 不把完整 HeaderGraph、W3C ontology、XBRL 数据库或更复杂 Judge 当成 RAG 必需前置。
- 本轮只核对、整理系统源码并提交 GitHub，**没有部署新镜像、迁移数据或新增模型调用**。
- 不新增测试、Gold、答案、PDF、模型权重、实验产物、简历或教学 HTML 到 GitHub。

## 2. 仓库与提交方式

- 远程：<https://github.com/yanovoUltra/newrag>，默认分支 `main`。
- 本轮开始时远程基线：`a3bd1a4b7e735d76a0279d30c5cef3eaabac19c9`。
- 本地原 `main`：`db7db78`，含 8 个未推送提交；这些提交混有实验资料，不能直接整体推送。
- 发布采用独立索引，从远程基线提取当前系统文件形成精简提交；发布分支为 `homelander/system-handoff-20260909`，推送到远程 `main`，不强推。
- 本地原 `main`、普通暂存区、历史未提交改动保留。**推送后本地 main 不等于远程 main，不要直接 pull/reset 或再次推送旧 main。** 新会话先运行 `git fetch origin`、`git log -1 origin/main`、`git status --short`。需要独立开发时，基于更新后的 `origin/main` 创建新 worktree，不要覆盖本工作区。
- 本轮不重写远程历史；远程原已存在的测试/历史文件不代表本轮新增，也没有进行批量删除。
- 发布范围含 API、权限、存储、解析、检索、回答、前端、Compose、运行依赖、无密钥示例和本交接文档。评测页面的汇总服务/静态摘要是运行依赖，不是本轮全量评测或最新成绩。

## 3. 已核实的实现及实际运行

| 层 | 当前源码能力 | 运行边界 |
|---|---|---|
| 上传与解析 | PDF/DOCX/XLSX/图片、阶段缓存、坏文本检测、按需 OCR、表格与章节分块 | OCR 当前关闭；不等于任意复杂表均可靠 |
| 检索 | dense/sparse/table/字段/数字短语/章节候选、RRF、Reranker；简单查询保护公司/年份/指标约束 | 不把历史检索分数当作端到端分数 |
| Relay | 使用 Qdrant 原始文件名、页码、章节；校验 doc/org/visibility | **本地修复尚未进入常驻 API 镜像** |
| 上下文 | 完整页候选与权限验证、预算控制、可选整块打包/去重 | 新整块打包和去重默认 false；metric/Relay 不走该分支 |
| 回答 | 主/轻量模型路由、SSE、引用、字段路径 | Decimal/citation 辅助模块存在，不据此宣称所有回答已接入确定性计算 |
| 数据与权限 | PostgreSQL 业务登记、Qdrant 向量、Redis 缓存与队列、OIDC/JWT 租户权限 | Keycloak 使用独立 PostgreSQL；不是跨库原子事务 |

当前快照：API/worker 为 `newrag-api:local-final-20260907`，frontend 为 `newrag-frontend:local-final-20260907`。容器健康；worker ping 返回 pong。回环前端 HTTP 200，API 容器内部 `/livez` 与 `/healthz` 均为 200。PowerShell 直接访问 localhost 的探测超时，改用不走代理的回环请求通过，不将代理探测失败写成服务故障。

运行模型：主 `deepseek-v4-flash-vision-exp`，轻量 `deepseek-v4-flash`；`runtime_profile=local-real`，业务库后端 `postgresql+psycopg`。这次只读取运行配置，不代表重新完成真实模型鉴权或质量验收。

Relay 源码 SHA-256（字节身份，不同换行也会影响 SHA；同时已有行为差异核对）：

- 本地 `backend/app/retrieval/search.py`：`89b3684d7d76eacf64bce4dad96f756d079562fb4808daaf64b72cfcf7c32465`。
- 容器 `/app/app/retrieval/search.py`：`a1d963c856e1039b512191a6e26af00c5de83ce07c22ad0b47de259a8fa6b1d2`。

另有 indicator168 和 r8 两个独立 Qdrant 实验容器，它们不是不同题目的业务服务，也没有被本次迁移、关闭或合并。

## 4. 168 题：最新回答结果，不能与其他题集混用

来源：本地 `data/experiments/xbrl_dual_source_dev_v1/rag_evidence_repair_v1/indicator168_relay_answer_successor_v1/RESULTS.md` 及 `final_comparison.json`，本轮已读取核对。

| 指标 | 历史基线 | 旧答案同期复核 | 修复后答案 |
|---|---:|---:|---:|
| 完整正确 | 146/168，86.90% | 148/168，88.10% | 149/168，88.69% |
| 忠实度 | 147/168，87.50% | 150/168，89.29% | 155/168，92.26% |
| 上下文内引用正确 | 144/168，85.71% | 145/168，86.31% | 152/168，90.48% |
| 三项联合 | 135/168，80.36% | 137/168，81.55% | 145/168，86.31% |

168 次重新回答、336 次旧新审核完成。同期净增 8 题、4.76pp；相对历史 135 多 10 题。原始模型给新版 146，来源复核后为 **145**，不能取更高值。最终 12 FAIL、11 UNKNOWN 均在分母。

这是 **冻结召回后重新回答与模型辅助审核**，不是 RAGAS、不是重跑 Embedding/Rerank、不是生产全链路或独立泛化验收。仍有错主体/错期间/口径、额外算术断言和审核误判；没有达到 0.95。

历史检索：P@1/NDCG@1 99.40%、MRR@8 99.60%；本地简历记录 Recall@8 85.99%。本轮未重新计算检索指标。原结果入口为 `indicator168_integrated_final_v2/evaluation/results.json`，不要用回答分数推断 Recall 提升。

非指标题历史汇总：62 题中 60 可回答题，GroupRecall@20 45.22%、All-Hit@20 16/60；这是 `PROJECT_SUMMARY.md` 所列历史基线，本轮未重评。后续上下文方案是否有稳定回答收益仍未闭环。四公司 80 题属于另一合同，不能替代这里的 168/62 题。

## 5. 下一步，按这个顺序继续

1. 先确认要做的是 **168 题真实回答改进**还是 **非指标题覆盖改进**；“继续”不得默认启动其他题集。
2. 优先处理 168 已暴露的期间/口径选择和多余计算断言；复用已保存失败证据，不能看分数反复改 Gold。
3. 如要接入常驻服务，单独确认发布范围，构建并核对镜像源码身份，再验证真实请求。GitHub 提交不等于部署。
4. 非指标题保留独立指标/分母，先检查来源互补和实际预算内保留，再决定是否晋级打包方案。
5. Judge 只改善测量可靠性，不直接改善答案。现有 Completeness Policy v2 的离线兼容回放不是新校准；不要因“继续”自动启动 354 审核或新一轮构造样本。

每次昂贵调用前确认：题集路径/题数、基线、修复版本、评测配置、分母、调用范围。先核验身份，再执行；不重复生成已完成答案、不默换模型。

## 6. 新会话文件入口

- 本文：当前目标、运行差异、结果边界、下一步。
- `backend/app/pipelines/answer.py` / `backend/app/retrieval/search.py`：实际问答与召回入口。
- `backend/app/core/identity.py` / `backend/app/api/v1/documents.py`：身份及来源访问权限。
- `backend/app/store/registry.py` / `backend/app/store/qdrant.py`：关系库与向量库。
- `infra/RUNTIME_PROFILES.md` / `infra/docker-compose.production.yml` / `infra/docker-compose.local-real.yml`：本地认证完整栈，名称 production 是兼容历史卷身份，**不是公网发布许可**。
- `.env.example`：公开配置模板；实际 `.env` 与 keyring/Docker secrets 只在本机读取，禁止输出或提交密钥。
- 本地 `PROJECT_SUMMARY.md`、`project_tutorial.html`：历史详细记录/教学；不是最新运行状态的唯一依据。
- 本地 `output/pdf/黄必彦_AI应用开发工程师.pdf`：简历，不是验收制品，未随本轮发布。

当前 GitHub 不含本机评测题目、答案和原 PDF；新机器克隆后不能假定可以直接复现私人数据评测。无模型/适配器/凭据时必须明确配置缺失，不假装服务可用。

## 7. 本轮检查

- 当前系统相关后端测试：129 passed（无真实模型调用，测试保留在本地）。
- 独立导出的发布快照：API、入库、回答、页证据、打包、本地 reranker 服务入口导入通过；90 个应用 Python 文件语法检查通过。
- 前端 `vue-tsc --noEmit -p tsconfig.app.json` 通过；没有无必要重复完整构建或模型评测。
- 选择文件的常见密钥格式扫描、应用 import 依赖检查、暂存差异空白检查通过；不是对仓库历史的全面安全认证。
- 发布差异中测试、题集、原始答案、PDF、`.env`、输出目录文件为 0；测试仍保留在本地并已执行。
- 未部署、未训练、未接触封存题目；原工作区和普通暂存区保留，不批量删除文件。
