---
name: newrag-full-lifecycle
description: "Deliver a financial-report RAG or competitor-analysis Agent end to end: understand an existing repository, audit performance and UX, improve retrieval and evaluation, harden ingestion and streaming, containerize and deploy behind HTTPS, migrate SQLite/files/Qdrant consistently, and verify production. Use for NewRAG-like FastAPI, Vue, Celery, Redis, Qdrant, SQLite, Docker Compose, financial-document RAG, retrieval benchmark, production deployment, disaster recovery, or local-to-server migration work."
---

# NewRAG Full Lifecycle

Turn an existing financial-document RAG into a measured, deployable, recoverable production system. Preserve compatibility and evidence; never claim success from a build alone.

## Select the track

- For review or planning only, inspect and report without changing code or external state.
- For implementation, establish a baseline, change one priority band at a time, and rerun proportional tests.
- For deployment, read [references/production-deployment.md](references/production-deployment.md).
- For data migration or recovery, read [references/data-migration.md](references/data-migration.md) completely before writing data.
- For retrieval metrics or reranker work, read [references/retrieval-evaluation.md](references/retrieval-evaluation.md).
- For final acceptance, use [references/acceptance-checklist.md](references/acceptance-checklist.md).

## Execute the lifecycle

### 1. Establish scope and safety

1. Read repository instructions and all project Markdown files relevant to architecture, setup, testing, deployment, and evaluation.
2. If `.codegraph/` exists, use CodeGraph before grep or broad file reading.
3. Record the repository root, active branch, dirty files, runtime versions, configured services, data locations, and protected production targets.
4. Treat credentials, private keys, model endpoints, organization identifiers, local paths, and financial questions as sensitive. Never print secret values or copy them into the skill, logs, commits, or reports.
5. Separate facts verified now from assumptions and earlier notes. Use explicit success criteria.

### 2. Build a factual baseline

Run the existing backend tests, frontend tests, production build, dependency checks, and service health probes. Measure before optimizing:

- API latency and dependency latency separately.
- frontend entry and route chunk sizes, gzip sizes, and rendering hot paths.
- Redis key families, memory, TTL, hit rate, and eviction policy.
- upload peak memory and duplicate lookup behavior.
- SQLite journal mode, busy timeout, indexes, and concurrency behavior.
- document, structured-field, file, pipeline-artifact, and vector counts.
- retrieval metrics on a frozen labeled dataset.

Do not silently relax failing tests or compare metrics from different query sets.

### 3. Prioritize by measured impact

Use P0 for correctness, data loss, security, severe memory growth, misleading limits, or broken health. Use P1 for latency, bundle size, upload memory, mobile/accessibility, indexes, and batching. Use P2 for decomposition, observability, generated types, linting, and deployment ergonomics.

For each item record: evidence, proposed change, risk, verification, expected effect, and compatibility impact.

### 4. Implement in reversible slices

Prefer this order:

1. correctness and public configuration contracts;
2. caching, health, upload limits, and streaming hot paths;
3. chunked I/O, batched embeddings/vector writes, and direct database queries;
4. frontend bundle, mobile UX, accessibility, and cancellation;
5. SQLite concurrency and indexes;
6. retrieval experiments and reranking;
7. module decomposition, observability, and deployment packaging.

Keep old API shapes working when practical. Add new endpoints before removing old ones. Rebuild and test after each slice so regressions have a small search area.

### 5. Evaluate retrieval honestly

Keep indicator queries and non-indicator/evidence-completeness queries separate. Use ranking metrics for the first and All-Hit@k plus coverage for the second. Optimize against a development split and report once on an untouched test split. Never promise a target such as 0.90 or 0.95 before measurement.

### 6. Productionize the system

Create reproducible backend, worker, frontend, datastore, and gateway definitions. Pin versions, run applications as non-root, isolate cache Redis from the broker, persist stateful volumes, add health checks and resource controls, and terminate trusted HTTPS at the gateway. Empty model API keys may allow infrastructure deployment, but model-dependent features must then report unavailable rather than fabricate output.

### 7. Migrate data as one consistency unit

Treat registry database, original uploads, pipeline artifacts, Qdrant collection, and structured-field evidence links as a single dataset. Quiesce writers, create source snapshots and destination rollback backups, verify hashes, restore all components, remove only explicitly identified test records, and perform bidirectional referential checks. Use `scripts/verify_newrag_consistency.py` for a read-only check when its schema assumptions match.

Never replace a live SQLite main file while stale `-wal` or `-shm` files remain. Never bulk-delete a Qdrant collection merely to remove a few test points.

### 8. Prove the user-visible outcome

Verify local and production independently:

- all required containers are healthy/running and the worker returns pong;
- frontend, liveness, readiness, document paging, config, and Agent endpoints respond;
- database document IDs and vector payload document IDs match both ways;
- structured evidence source chunk IDs resolve to vectors;
- a representative competitor analysis produces available values and evidence;
- recent logs contain no new exception loop or restart loop;
- backups and a tested rollback path remain available.

Report actual counts, endpoints, test results, excluded test data, known limitations, and rollback locations. Do not expose authentication secrets in the handoff.

## Maintain operational discipline

- Inspect exact targets before stopping, replacing, deleting, or moving state.
- Do not terminate unrelated processes to resolve port conflicts.
- Preserve user changes and unrelated dirty files.
- Prefer server-side time as the final authority for date and fiscal-year validation.
- Use mock/local model services for performance tests unless external evaluation is explicitly authorized.
- When external reranking is authorized, cap spend and log dataset/config/version rather than sensitive content.
- Keep the system recoverable at every stage; a migration without rollback evidence is incomplete.
