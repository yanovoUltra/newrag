# Production deployment

## Contents

1. Preflight
2. Compose topology
3. Secure exposure
4. Rollout and rollback
5. Verification

## Preflight

Confirm server identity, SSH user/key, OS/architecture, disk and memory headroom, Docker daemon, DNS or public IP, security-group ports, existing listeners, and data-retention requirements. Inspect ports before starting; stop on unrelated conflicts.

Never print `.env`, private keys, passwords, or model credentials. Transfer secrets through protected environment files or a secret manager. Set restrictive permissions.

## Compose topology

Provide independent services for:

- API and Celery worker built from the same pinned backend artifact;
- frontend static server;
- Qdrant persistent volume;
- cache Redis with bounded memory and `allkeys-lru`;
- broker Redis with `noeviction`;
- HTTPS gateway such as Caddy or Nginx.

Use non-root application users, narrow build contexts, explicit health checks, restart policies, named volumes, and resource limits. Keep optional observability profiles disabled by default.

## Secure exposure

Expose only the gateway. Prefer a trusted domain certificate. If serving an IP, use a certificate chain trusted by the intended clients or clearly document trust installation; do not call a self-signed warning “trusted HTTPS.” Apply security headers and protect administrative or private systems with an identity layer. Rotate any credential exposed in logs or chat.

If model API values are intentionally empty, deploy infrastructure but make chat/rerank behavior explicit and graceful.

## Rollout and rollback

1. Create backups of current code/configuration and every stateful volume.
2. Build images before stopping the old service when possible.
3. Start datastores, then API, worker, frontend, and gateway.
4. Wait for health instead of sleeping a fixed duration.
5. Run smoke tests and inspect recent logs.
6. Keep old images and backups until acceptance completes.

Do not run bulk cleanup commands after a failed deployment. Preserve containers and logs for diagnosis.

## Verification

Verify frontend HTTP status, `/livez`, `/healthz`, public config, document paging, worker `inspect ping`, Redis policies, Qdrant collection count, TLS chain, current container health, restart count, and recent logs. Report image sizes and actual endpoints.
