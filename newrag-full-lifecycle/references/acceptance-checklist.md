# Acceptance checklist

## Code and tests

- Repository instructions and relevant Markdown were read.
- Baseline and final backend/frontend tests are recorded.
- Production frontend build succeeds with measured bundle sizes.
- Dependency/security checks are recorded without overstating coverage.
- API compatibility and fiscal-year boundary tests pass.

## Performance and UX

- Cached health P95 and dependency timeout behavior are measured.
- Upload limit displayed by the UI equals the server limit.
- Uploads are bounded-memory and stop early when oversized.
- streaming updates, storage writes, and scroll work are throttled.
- mobile widths have no horizontal overflow and navigation does not wrap.
- labels, keyboard operation, status announcements, and 44px targets are checked.

## Retrieval and Agent

- corpus, labels, split, model, reranker, `k`, and commit are frozen.
- indicator metrics and non-indicator All-Hit@k are reported separately.
- citations resolve to existing chunks and permitted documents.
- competitor analysis reports missing/conflicting/non-comparable cells honestly.
- external model/rerank cost, timeout, and fallback behavior are known.

## Infrastructure

- frontend, API, worker, Qdrant, cache Redis, broker Redis, and gateway are healthy.
- cache and broker eviction policies are correct.
- only intended ports are public.
- HTTPS trust and authentication were tested from a client.
- current logs show no exception/restart loop.

## Data

- source and destination manifests and hashes exist.
- pre-migration destination rollback backups exist.
- SQLite integrity, document/field counts, file counts, and vector counts match.
- DB-to-vector, vector-to-DB, and evidence-to-vector differences are empty.
- test data exclusions are documented.
- one user-visible document listing and one evidence-bearing analysis succeed.

## Handoff

- Report facts, counts, endpoints, caveats, and rollback locations.
- Do not publish passwords, keys, tokens, or private model configuration.
- Distinguish completed work from recommendations and unavailable model features.
- Leave backups until the user explicitly approves retirement.
