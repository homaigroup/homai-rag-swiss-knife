# Feature Matrix — v4.0.0

| Capability | Status | Verification |
|---|---|---|
| Persian/Arabic normalization | production core | tested |
| Normalized ↔ original offsets | production core | tested |
| Approximate tokenizer | production core | tested |
| HF/tiktoken exact token counters | optional adapters | contract/syntax; runtime package absent |
| Model-aware chunk budget clamp | production core | tested |
| Deterministic duplicate-safe IDs | production core | tested |
| Idempotent/incremental source replacement | production core | tested |
| Atomic SQLite source replacement | production core | tested |
| Atomic source deletion + feedback cleanup | production core | tested |
| Composite mixed-document chunking | production core | tested |
| Adaptive semantic chunking | production core | tested |
| General overlap policy | production core | tested |
| Table-aware chunking | production core | tested |
| Python AST code chunking | production core | tested |
| Tree-sitter parser hook | optional adapter | injected-parser contract |
| Parent-child records | production core | tested |
| Query-aware context expansion | production core | tested |
| Page/layout provenance | production core | tested |
| PDF loader + OCR hook | optional documents extra | live-tested with pypdf/reportlab |
| DOCX headings/tables | optional documents extra | live-tested with python-docx |
| HTML hidden-content exclusion | production core | tested |
| Multi-representation dense candidates | production core | tested |
| BM25F field-aware lexical index | production core | tested |
| Exact identifier/reference index | production core | tested |
| Sparse/SPLADE path | optional adapter | callable + fake SparseEncoder tested |
| Weighted RRF | production core | tested |
| No-evidence empty-result invariant | production core | tested |
| Query routing/decomposition | production core | tested |
| True token-matrix late interaction contract | optional stage | tested with callable encoder |
| Cross encoder | optional adapter | activation contract tested; live model absent |
| MMR/source caps | production core | tested |
| HNSW named-vector ANN | optional adapter | filter-aware contract tested; hnswlib absent |
| Matryoshka HNSW cascade | optional adapter | contract-tested |
| Qdrant named dense + sparse backend | production backend adapter | fake-client integration tested; qdrant-client/server absent |
| Generic external backend interface | adapter | tested |
| Graph local/global/DRIFT-lite | optional local graph | tested |
| Visual candidate fusion | optional adapter | tested |
| Multi-vector visual MaxSim | optional adapter | tested |
| Matryoshka truncation/int8 helpers | utility | tested |
| SQLite persistence/restart | production local store | tested |
| Async facade | production core | tested with SQLite + concurrency |
| Embedding/query caches | production core | tested |
| ACL/multi-tenancy | production core | adversarial tests |
| Security scanning/source trust | defense-in-depth | adversarial tests |
| Tracing/reasons | production core | tested |
| OpenTelemetry trace sink | optional adapter | live API smoke-tested |
| Durable feedback | production local store | tested |
| Pointwise + pairwise linear rankers | built-in lightweight models | tested |
| Retrieval evaluation | production tooling | tested |
| Chunk boundary/citation helpers | tooling | tested |
| Chunking policy optimizer | tooling | tested |
| Score calibration hook | optional | tested |
