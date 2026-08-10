# NewRAG 前端

多模态财报深度分析 RAG 系统的 Web 前端（阶段三）。基于 Vue 3 + TypeScript + Vite + Element Plus，提供文档上传、文档库管理与流式问答三大页面。

## 功能

- **智能问答**（`/chat`）：SSE 流式问答，Markdown 渲染 + 引用溯源回跳，问答记录持久化。
- **文档库**（`/documents`）：文档列表、状态查看、归档/版本控制。
- **上传文档**（`/upload`）：拖拽上传 + 任务状态轮询（parse → chunk → embed → index）。
- **服务健康检查**：侧栏实时显示后端 `/healthz` 状态。

## 技术栈

| 分类 | 选型 |
| --- | --- |
| 框架 | Vue 3（`<script setup>`）+ Vue Router |
| 构建 | Vite + TypeScript + vue-tsc |
| UI | Element Plus + `@element-plus/icons-vue` |
| 渲染 | marked + dompurify（安全渲染 Markdown） |
| 测试 | Vitest（单测）+ Playwright（E2E） |

## 开发

```bash
npm install
npm run dev       # 启动开发服务器，默认 http://localhost:5173
```

`vite.config.ts` 已配置代理：`/api` 与 `/healthz` 转发到 `http://localhost:8000`（后端）。需先启动后端服务。

## 构建

```bash
npm run build     # vue-tsc 类型检查 + vite build，产物输出到 dist/
npm run preview   # 本地预览构建产物
```

## 测试

```bash
npm test          # Vitest 单测（如 SSE 解析）
npm run test:e2e  # Playwright 端到端（需后端 8000 运行）
```

## 目录结构

```text
frontend/
├── src/
│   ├── views/          # ChatView / DocumentsView / UploadView
│   ├── api/            # sse（fetch+ReadableStream）/ documents / chat
│   ├── components/     # ChatMessage / MarkdownText
│   ├── router/         # 路由（/chat /documents /upload）
│   ├── styles/theme.css # 设计令牌（AI-Native 暗色）
│   ├── App.vue / main.ts / types.ts
├── e2e/                # Playwright 端到端（chat.spec.ts）
├── playwright.config.ts
├── vite.config.ts      # dev 代理 /api → http://localhost:8000
└── package.json
```