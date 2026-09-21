# Migration: 3.x -> 4.0

The common API remains compatible:

```python
engine = SemanticChunkSearch(ChunkingConfig(...))
engine.ingest(text, source_id, metadata=...)
result = engine.search(query, top_k=5)
```

## Important behavior changes

1. Re-ingesting identical text now still applies changed ACL/tenant/metadata state.
2. Identical chunks in the same section receive distinct deterministic IDs instead of overwriting each other.
3. Search returns an empty result when all retrieval stages have no evidence; it no longer injects arbitrary fallback candidates.
4. Trust/feedback policy is applied after reranking, so a custom reranker cannot erase source trust.
5. Context packing always reserves the retrieved hit before adding parent/neighbors.
6. `ingest_pages()` runs the same security scan as normal ingestion and skips blank pages safely.
7. SQLite uses thread-safe WAL access and atomic source replacement/deletion.
8. Source-level evaluation deduplicates sources before NDCG/Recall calculations.
9. Local lexical retrieval is field-aware BM25F.
10. Local HNSW uses ANN with filtering instead of falling back to a full exact scan for ordinary security/metadata filtering.
11. Chunk token budgets can be clamped to the active embedding model's maximum input length.
12. CrossEncoder activation is explicit (`model`, `sigmoid`, or `identity`).

## Persistence

Existing v3 JSON payload rows are loaded with forward/backward-compatible field filtering. Changed embedding/vectorizer signatures trigger re-vectorization when required. For a major production migration, back up the SQLite database and benchmark retrieval quality before switching traffic.

## Production backend

For large corpora prefer `QdrantSemanticIndex` or another `BaseSemanticIndex` implementation. Avoid backend implementations that require enumerating an entire collection for source updates; implement source-scoped IDs/delete/upsert/filter operations.

## Optional extras

- `neural`: sentence-transformers dense embeddings
- `ann`: hnswlib
- `reranker`: sentence-transformers cross encoder
- `tokenizers`: transformers + tiktoken
- `sparse`: sentence-transformers SparseEncoder/SPLADE
- `qdrant`: qdrant-client
- `documents`: pypdf + python-docx
- `observability`: opentelemetry-api
- `all`: all optional integrations
