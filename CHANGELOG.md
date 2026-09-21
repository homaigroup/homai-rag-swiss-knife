# Changelog

## 4.0.0

Hardening release focused on correctness, security, recovery, and deployability.

- Fixed ACL/metadata re-ingestion when source text is unchanged.
- Fixed duplicate-chunk stable-ID collisions.
- Fixed async SQLite cross-thread failures with a locked WAL store.
- Added atomic SQLite source replacement and durable source-delete cascade.
- Added rollback recovery for failed/partial source updates and deletions.
- Made feedback durable-first and persistence-aware.
- Fixed page-ingestion security scanning, blank-page handling and source-wide hashing.
- Added HTML hidden-DOM exclusion.
- Rebuilt mixed-section parent text and protected hit inclusion during context packing.
- Moved trust/policy enforcement after learned/callable rerankers.
- Removed arbitrary-candidate fallback when retrieval has no evidence.
- Reworked local lexical retrieval into field-aware BM25F.
- Reworked local HNSW into named-representation, filter-aware ANN; added Matryoshka coarse-to-full rescoring.
- Added model/tokenizer-aware chunk budget clamping and stronger embedding/vectorizer signatures.
- Added multilingual cross-encoder default and explicit activation contract.
- Added real Qdrant backend contract with named dense vectors, optional sparse vector, ACL payload filtering and source-scoped operations.
- Added pairwise linear ranking, model-confidence routing and more robust prompt-injection heuristics.
- Added PDF/DOCX/HTML ingestion improvements, OCR hook and layout provenance.
- Added OpenTelemetry trace sink, optional-stage fail-open diagnostics and strict mode.
- Fixed source-level NDCG and hard-negative evaluation helpers.
- Expanded regression suite to 63 tests.

## 3.0.0

- Added multilingual/Persian normalization and tokenizer abstraction.
- Added deterministic IDs, idempotent/incremental ingestion and version metadata.
- Added composite, table-aware, adaptive semantic, AST-aware and Tree-sitter-hook chunking.
- Added multi-representation candidate generation, inverted BM25, exact IDs, sparse and graph candidates.
- Added weighted RRF, reranking, MMR, parent/context expansion, persistence, ACL/security, multimodal hooks and evaluation.
