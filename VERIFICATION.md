# Verification — v4.0.0

Verification date: 2026-09-14

## Release gates

- `pytest`: **63 passed, 0 failed**.
- Statement coverage: **88% overall**; central pipeline/search/context/security modules are materially higher.
- `compileall`: passes for package and tests.
- Offline wheel build: passes with `pip wheel --no-build-isolation --no-deps`.
- Clean virtual-environment install: passes.
- `pip check`: passes with no broken mandatory requirements.
- Installed-wheel smoke test: passes for Persian mixed-document ingestion, private ACL enforcement, ACL change on unchanged text, exact-ID hybrid retrieval, context expansion, SQLite restart persistence, and async ingest/search.
- Installed-wheel concurrency stress: **80 async writes + 160 async searches** completed without an exception; restart after stress preserved the final source state.
- Real optional-runtime tests available in this environment: `pypdf`, `python-docx`, `reportlab`, and OpenTelemetry API all pass live tests.

## Hardened regression gates

The suite explicitly covers the failure modes found during the v3 audit:

- ACL/metadata updates on re-ingest with unchanged text.
- Default tenant isolation.
- Deterministic duplicate-safe chunk IDs.
- Async + SQLite thread safety.
- Page-ingestion security scanning and blank-page handling.
- Full-section parent reconstruction for mixed prose/code.
- Context-budget invariant: retrieved hit cannot be displaced by parent/neighbor context.
- Source-trust policy applied after reranking.
- Atomic source replacement and rollback on persistence failure.
- Atomic durable source deletion and rollback after a partial in-memory deletion failure.
- Durable-first feedback recording.
- Original/normalized offset mapping.
- Source-level NDCG bounded to `[0,1]` and deduplicated.
- Embedding-model token-budget clamping.
- Cross-encoder activation contract (no implicit double sigmoid).
- Filter-aware HNSW path remains ANN rather than exact-scan fallback (contract-tested with a fake hnswlib runtime).
- Main-pipeline multimodal candidate fusion.
- True token-matrix late-interaction encoder contract.
- Qdrant named-vector/sparse/ACL/source-replace contract with a fake client.
- Source-wide hash stability for paged ingestion and layout provenance retention.
- Model-based query-routing confidence fallback.
- Pairwise ranking preference learning.
- Zero-width/base64 prompt-injection detection.
- Embedding/contextualizer signature invalidation and re-embedding.
- Real DOCX heading/table extraction.
- Real PDF extraction and OCR fallback.
- OpenTelemetry sink smoke test.
- Optional-stage fail-open diagnostics and strict failure mode.
- SparseEncoder API contract.
- No-evidence search never manufactures arbitrary candidates.
- HTML hidden DOM content is excluded from ingestion.

## Optional runtimes not present in this execution environment

The following third-party packages were not installed locally and therefore were not live-model/server tested here:

- `hnswlib`
- `sentence-transformers`
- `qdrant-client`
- `transformers`
- `tiktoken`

Their package integration contracts are covered with fake/injected implementations where applicable. This is intentionally reported as **contract-tested**, not live-tested. A production deployment should run the included test suite again with the exact optional extras and model revisions used in that environment.

## Scale boundary

`InMemorySemanticIndex` and SQLite persistence are hardened local/small-corpus components, not substitutes for a distributed vector/search service. For large corpora use `QdrantSemanticIndex` or another `BaseSemanticIndex` backend that implements persistent named vectors, sparse search, native filtering, batch operations, and ANN.
