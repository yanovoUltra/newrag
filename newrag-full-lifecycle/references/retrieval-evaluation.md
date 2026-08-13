# Retrieval evaluation

## Contents

1. Dataset design
2. Metric selection
3. Experiment loop
4. Production acceptance

## Dataset design

Freeze query IDs, corpus version, relevance judgments, eligible organizations, fiscal periods, and visibility rules. Split by document/company or question family to reduce leakage. Keep an untouched test set.

Maintain at least two query classes:

- **Indicator queries**: a single financial fact, metric, company, period, and evidence location are rankable.
- **Non-indicator queries**: comparisons, causes, risks, trends, or summaries require several evidence groups.

For every run persist: code commit, index snapshot, embedding model, sparse configuration, fusion weights, candidate depth, reranker/version, prompt/query expansion, and latency.

## Metric selection

For indicator queries report `NDCG@k`, `Recall@k`, `Precision@k`, `MRR`, answerable rate, and evidence correctness. Do not optimize only one metric.

For non-indicator queries define required evidence groups per query and use:

`All-Hit@k = queries whose top-k contains at least one relevant chunk from every required group / all eligible queries`

Also report group coverage, evidence precision, citation correctness, latency, and token/cost impact. All-Hit@k is appropriate for completeness, but it must not replace precision or answer-quality checks.

Never compare `@8` with another `k` without labeling the change. Choose the smallest operationally affordable `k` that reaches the completeness target.

## Experiment loop

1. Reproduce the baseline deterministically.
2. Inspect misses by category: parsing/chunking, metadata, entity/period normalization, sparse recall, dense recall, fusion, reranking, or labeling error.
3. Fix the earliest failing stage instead of compensating downstream.
4. Test one controlled change at a time.
5. Tune only on development data; run the final configuration once on test data.
6. Record confidence intervals or per-query deltas when the set is small.

High-yield levers include table-aware chunks, section paths, company aliases, fiscal-period normalization, metric synonyms, numeric phrase indexes, hybrid candidate union, reciprocal-rank fusion, metadata filters, diversified evidence groups, and a finance-domain reranker.

External rerank APIs may be used only when authorized. Use bounded candidate counts, timeouts, retries, caching, cost measurement, and a deterministic fallback. Never send secrets or unnecessary document text.

## Production acceptance

Do not declare “production grade” solely because a metric exceeds 0.90. Require stable latency, visibility isolation, citation correctness, failure fallback, drift monitoring, and a frozen reproducible evaluation artifact. Report ceilings honestly when label ambiguity or reranker quality blocks further gains.
