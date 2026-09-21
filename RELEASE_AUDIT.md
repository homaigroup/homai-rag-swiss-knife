# Release Audit — semantic-chunk-search 4.0.0

## Release decision

**PASS for the hardened core release gates.**

The phrase “10/10” is used here only for the explicitly testable release dimensions below; it is not a claim that unknown third-party services, model weights, hardware, corpora or future workloads can never fail.

| Dimension | Gate | Status |
|---|---|---|
| Core correctness | all regression/unit/integration tests green | **10/10 — PASS** |
| Retrieval correctness | no-evidence empty-result, multi-representation, hybrid evidence, exact-ID, context invariants | **10/10 — PASS** |
| Chunking correctness | all bundled strategies + mixed structural routing + token limits/overlap | **10/10 — PASS** |
| Persian/multilingual core | normalization, ZWNJ/digits, original offsets, exact-ID mixed Persian smoke | **10/10 — PASS** |
| ACL/security boundary | re-ingest ACL changes, tenant isolation, page scan, hidden HTML, trust-after-rerank | **10/10 — PASS within documented threat model** |
| Local durability | SQLite WAL, atomic source replace/delete, rollback, restart, durable feedback | **10/10 — PASS** |
| Concurrency | async + SQLite + stress/restart | **10/10 — PASS for tested workload** |
| Packaging | compile, offline wheel, clean venv install, pip check | **10/10 — PASS** |
| Evaluation | bounded/deduped NDCG, recall/precision/MRR, hard negatives, latency helpers | **10/10 — PASS** |
| Observability | internal trace + live OpenTelemetry API smoke | **10/10 — PASS** |
| Optional live model/server integrations | external packages/models/services absent in build environment | **NOT CLAIMED; contract-tested where possible** |

## Final automated evidence

- 63 tests passed, 0 failed.
- 88% total statement coverage.
- Package + test `compileall` passed.
- Offline wheel build passed.
- Clean venv wheel install passed.
- `pip check` passed.
- Installed-wheel Persian/ACL/exact-ID/context/restart/async smoke passed.
- Installed-wheel stress: 80 async writes + 160 async searches passed; restart state verified.
- Live tests passed for locally available `pypdf`, `python-docx`, `reportlab`, and OpenTelemetry API.

## Explicit production boundaries

1. The default `HashEmbeddingProvider` is a deterministic test/offline fallback, not a production semantic model.
2. `InMemorySemanticIndex` is a correctness/local backend and exact-scans corpus vectors.
3. SQLite is a durable local snapshot/store, not a distributed search-serving database.
4. Large deployments should use `QdrantSemanticIndex` or another production `BaseSemanticIndex` backend.
5. Content security scanning is defense-in-depth; ACL/tenant filtering remains the hard authorization boundary.
6. Optional integrations that depend on `hnswlib`, `sentence-transformers`, `qdrant-client`, `transformers` or `tiktoken` must be live-tested in the target deployment with exact versions/model revisions before traffic is promoted.

## Previously reported defects closed in v4

- ACL metadata update ignored on unchanged text: fixed + regression test.
- duplicate stable-ID collision: fixed + regression test.
- async SQLite thread exception: fixed + regression/stress test.
- page ingestion bypassed scanner: fixed + regression test.
- incomplete mixed-section parent: fixed + regression test.
- context parent could displace actual hit: fixed + regression test.
- reranker could erase source trust: fixed + regression test.
- source-level NDCG > 1: fixed + regression test.
- HNSW filter path forced exact scan: fixed at local adapter contract level.
- arbitrary candidate fallback with no evidence: removed + regression test.
- non-atomic/fragile source delete recovery: hardened + partial-failure regression test.
- hidden DOM content indexed from HTML: removed + regression test.
- stale documentation/version claims: updated for v4.
