# Architecture — v4.0.0

## Ingestion

```text
Source / file / pages
  -> loader + provenance
  -> Unicode/Persian normalization + original-offset map
  -> security scan + ACL/tenant metadata + version/signatures
  -> structural/composite chunking
       -> prose recursive/semantic
       -> Markdown hierarchy
       -> table-aware
       -> Python AST / Tree-sitter hook
  -> overlap + duplicate-safe deterministic identity
  -> parent/neighbor graph
  -> enrichment/contextualization
  -> raw / summary / keywords / titles / references
  -> batched embeddings + BM25F + exact IDs + optional sparse + graph
  -> atomic source replacement
  -> optional durable SQLite snapshot or external index backend
```

## Query

```text
Query
  -> normalize / route / optional decomposition
  -> query representations
  -> named-vector dense candidates
  -> BM25F candidates
  -> optional sparse candidates
  -> exact/reference candidates
  -> optional graph/visual candidates
  -> weighted RRF
  -> multi-representation evidence score
  -> optional late interaction
  -> optional cross encoder / LTR
  -> hard trust/policy layer
  -> optional score calibration
  -> threshold + MMR/source cap
  -> hit-first parent/neighbor expansion under token budget
```

## Security boundary

ACL/tenant checks are part of candidate retrieval, not a presentation-time filter. `source_trust` and feedback cannot be overwritten by reranker output. The content scanner is defense-in-depth only; the hard boundary is permission-aware retrieval and tenant policy.

## Recovery boundary

Source replacement is atomic inside the bundled in-memory index and transactionally persisted in SQLite. On a failure after a partial update, the facade restores the known-good source snapshot. SQLite uses WAL, full synchronous mode and locked cross-thread access.

## Scale boundary

- `InMemorySemanticIndex`: exact, thread-safe, intended for local/small corpora and correctness testing.
- `HNSWSemanticIndex`: optional local ANN over all named dense representations with filtered HNSW traversal.
- `MatryoshkaHNSWSemanticIndex`: optional coarse ANN + full-vector rescore for MRL embeddings.
- `QdrantSemanticIndex`: external persistent named-vector/sparse backend adapter with source-scoped operations and ACL payload filtering.
- `CallableSemanticIndex`: generic integration surface for another production backend.

SQLite is a durable local document/vector snapshot, not a distributed serving database.
