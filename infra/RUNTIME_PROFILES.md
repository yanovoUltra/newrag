# NewRAG 运行配置合同

本文件定义应用运行时的单一配置口径。Parser P4 benchmark 使用自己的冻结 manifest，不能被这里的模型配置覆盖。

## 配置优先级

后端密钥解析顺序固定为：显式环境变量 → `*_FILE` Docker Secret → Windows/Linux OS keyring。容器不能读取宿主机 keyring，因此容器化真实运行必须使用 Docker Secret，不能因为宿主机 keyring 有值就宣称容器可用。

`MODEL_RUNTIME_MODE` 只表示 `demo` 或 `real` 能力合同；`RUNTIME_PROFILE` 表示本地运行场景。正式启动必须显式选择以下两种 profile：

| Profile | Embedding | LLM | Reranker | OCR | Mock | 用途 |
|---|---|---|---|---|---|---|
| `local-real` | 真实 | 真实 | 必须真实且健康 | 可选；启用后必须有 Token | 禁止 | 本机真实开发与质量验证 |
| `local-parser-benchmark` | 关闭 | 关闭 | 关闭 | 关闭 | 允许 | 冻结 Parser 实验；不运行 RAG |

留空 `RUNTIME_PROFILE` 只保留 `legacy-demo` / `legacy-real` 兼容标识，不属于可审计本地运行。

## 状态语义

运行配置把模型状态拆成三层，禁止再把“写了配置”说成“服务可用”：

- `configured`：后端类型、端点及所需凭据已声明。
- `reachable`：非计费 TCP/TLS 探测可以建立连接。
- `healthy`：服务提供了明确健康接口并返回正常；当前主要用于本地 reranker `/healthz`。

外部 Embedding/LLM/OCR 供应商没有统一、免费的应用健康接口，因此最多标记到 `reachable`；这不等价于模型调用、套餐额度或具体模型权限已经成功。

## 本机真实运行

本机 `.env` 使用 `local-real`。Qwen3-Reranker-8B 由独立脚本管理，只绑定 `127.0.0.1:8010`：

```powershell
.\infra\scripts\manage_local_reranker.ps1 -Action install
.\infra\scripts\manage_local_reranker.ps1 -Action start
.\infra\scripts\manage_local_reranker.ps1 -Action status
.\infra\scripts\manage_local_reranker.ps1 -Action stop
```

`install` 注册普通用户级 Windows 计划任务 `NewRAG Local Reranker`；任务持有前台模型进程并在异常退出后重启，避免模型随启动终端或 Codex 会话结束。`start` 在任务缺失时也会先注册，且只有 `/healthz` 返回正常才算成功。任务不申请管理员权限，服务也不会监听非回环地址。

默认加载：

- 基座：`D:\newrag-reranker\models\Qwen3-Reranker-8B`
- 指标题：`qwen3-reranker-8b-cover-all-positive-lr5e5\final-adapter`
- 非指标题：`final-non-indicator-v2\final-adapter`

启动门禁（不会打印密钥，也不会发起计费模型请求）：

```powershell
cd backend
$env:PYTHONPATH='.'
.\.venv\Scripts\python.exe scripts\runtime_config_gate.py --probe
```

如果 API/worker 也放进 Docker，必须同时使用 `infra/docker-compose.production.yml` 和 `infra/docker-compose.local-real.yml`，从当前 shell 注入 `NEWRAG_EMBEDDING_API_KEY`、`NEWRAG_LLM_API_KEY` 及完整栈要求的三项数据库/管理凭据。凭据从 OS keyring 获取，仅通过 Docker Secret 注入，不写入项目文件。不要单独使用基础 Compose 重建 API/worker，否则其 model-free 默认值会重新生效。

2026-09-08 常驻 API/worker 已改为：主模型 `deepseek-v4-flash-vision-exp`（视觉实验版），轻量模型 `deepseek-v4-flash`（官方当前0731版）。两型号均已通过真实文本请求，API/worker健康、worker pong已核验。此次未验证图片问答质量，不启用OCR，不修改历史Pro评测。embedding及reranker保持原配置。

2026-09-07 历史切换记录：embedding 为阿里云 qwen3.7-text-embedding（1024 维及 sparse），当时主模型 DeepSeek deepseek-v4-pro，轻量模型 deepseek-v4-flash；现有客户端在主模型 403 时可明确记录后切换 Flash，不回退 Mock。两模型和 embedding 均已实际调用通过。本地 reranker 两个 profile 均已实际调用通过，override 使用 `host.docker.internal:8010`，不使用容器自身的 `127.0.0.1`。OCR 仍关闭。
