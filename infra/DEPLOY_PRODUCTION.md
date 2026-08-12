# NewRAG 生产部署

生产编排文件是 `infra/docker-compose.production.yml`。它只向公网发布前端的 80 端口；API、Qdrant、缓存 Redis 和 broker Redis 只在 Docker 内部网络通信。

## 服务器要求

- Ubuntu 22.04/24.04 或同级 Linux。
- 最低 4 核、8 GB 内存、80 GB SSD；处理大型 PDF 建议 8 核、16 GB。
- 安全组入方向放行 TCP 22、TCP 80 和 TCP 443。
- 安装 Docker Engine 与 Docker Compose v2。

## 首次部署

1. 将项目上传到服务器，例如 `~/newrag`。
2. 将本地 `.env` 安全复制到服务器的项目根目录。不要提交到 Git。
3. 在 `infra/secrets/newrag.htpasswd` 配置入口 Basic Auth 凭据，并将文件权限限制为仅管理员可读。
4. 默认演示模式不需要外部 API 密钥；若切换真实模型，再在服务器端安全配置相应密钥。
5. 执行：

   ```bash
   cd ~/newrag
   docker compose -f infra/docker-compose.production.yml config --quiet
   docker compose -f infra/docker-compose.production.yml up -d --build
   docker compose -f infra/docker-compose.production.yml ps
   ```

6. 验证：

   ```bash
   curl --fail http://127.0.0.1/livez
   curl --fail http://127.0.0.1/healthz
   docker compose -f infra/docker-compose.production.yml exec -T worker \
     celery -A app.tasks.celery_app inspect ping --timeout=5
   ```

## 更新与回滚

更新不会主动删除现有卷或数据：

```bash
git pull --ff-only
docker compose -f infra/docker-compose.production.yml up -d --build
```

回滚时切回已验证的 Git 提交并再次执行 `up -d --build`。不要执行 `down -v`，否则会删除持久卷。

## 数据与备份

必须备份以下 Docker 卷：`newrag_app_data`、`newrag_qdrant_data`、`newrag_redis_broker_data`。缓存 Redis 可以重建，但业务高峰期仍建议保留。备份前应暂停上传与入库任务，避免 SQLite 与 Qdrant 快照时间点不一致。

## HTTPS

默认可信地址为 `https://8.134.222.143`。Certbot 使用 Let's Encrypt `shortlived` 配置签发约 6 天有效的公网 IP 证书，Caddy负责 TLS 终止与 HTTP 到 HTTPS 的永久重定向。`newrag-cert-renew.timer` 每 12 小时检查续期，并在续期后让 Caddy重新加载证书；证书状态保存在 `newrag_certbot_certs` 卷。对公网只公开 80/443。

## 生产限制

当前注册表使用 SQLite，适合单机单 API 实例。若要运行多个 API/worker 节点或要求高可用，需要迁移 PostgreSQL、对象存储和托管/集群化 Qdrant/Redis，并建立自动快照、监控告警与灾难恢复演练。

当前生产 Compose 默认使用无外部费用的演示模式：Mock Embedding、Mock LLM、RRF 直出。需要真实回答时，再在安全的服务器环境变量中配置模型密钥，并将 `EMBEDDING_BACKEND`、`LLM_PROVIDER`、`RERANK_BACKEND` 调整为对应真实后端。
