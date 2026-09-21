# semantic-chunk-search 4.0

Standalone multilingual chunking + hardened hybrid retrieval core for RAG/search systems.

The core has **no mandatory third-party runtime dependency**. Neural embeddings, ANN, cross-encoders, SPLADE, Qdrant, model tokenizers, PDF/DOCX and OpenTelemetry are optional extras.

## What v4 focuses on

v4 is a hardening release. It keeps the v3 architecture but closes correctness, security, rollback, persistence and deployment gaps found by adversarial audit.

- Persian/Arabic Unicode, digit and ZWNJ normalization with original-offset mapping.
- Model-aware token budgets and tokenizer abstraction.
- Duplicate-safe deterministic chunk IDs and source-scoped incremental replacement.
- Composite chunking for mixed prose, Markdown, code and tables.
- Adaptive semantic chunking, overlap policy, table-aware chunks and Python AST chunking.
- Parent/child hierarchy, neighbor links and hit-first token-budget context assembly.
- Multi-representation retrieval over raw text, summary, keywords, titles and references.
- Dense + field-aware BM25F + optional sparse/SPLADE + exact-ID + graph/visual candidates.
- Weighted RRF, multi-vector evidence scoring, optional late interaction/cross-encoder/LTR, MMR and source caps.
- ACL / tenant filtering during candidate retrieval; source trust is enforced after reranking.
- SQLite WAL persistence with atomic source replacement/deletion and rollback recovery.
- Async facade, batched embedding/sparse encoding, caching, diagnostics and OpenTelemetry hook.
- Real Qdrant backend adapter with named dense vectors, optional sparse vectors, payload ACL filters and source-scoped operations.
- PDF, DOCX, HTML and page ingestion with provenance; OCR is injectable.
- Retrieval/evaluation utilities: Precision/Recall@K, MRR, bounded NDCG, candidate recall, latency, boundary F1, citation coverage and chunking-policy optimization.

See `VERIFICATION.md` for exact release gates and optional runtimes that were or were not live-tested in the build environment.

## Install

Core:

```bash
pip install semantic-chunk-search
```

Optional capabilities:

```bash
pip install 'semantic-chunk-search[neural]'
pip install 'semantic-chunk-search[ann]'
pip install 'semantic-chunk-search[reranker]'
pip install 'semantic-chunk-search[tokenizers]'
pip install 'semantic-chunk-search[sparse]'
pip install 'semantic-chunk-search[qdrant]'
pip install 'semantic-chunk-search[documents]'
pip install 'semantic-chunk-search[observability]'
pip install 'semantic-chunk-search[all]'
```

## Quick start

```python
from semantic_chunk_search import ChunkingConfig, RetrievalConfig, SemanticChunkSearch

engine = SemanticChunkSearch(
    ChunkingConfig(
        strategy="auto",
        max_chunk_tokens=512,
        min_chunk_tokens=80,
        overlap_tokens=48,
        adaptive_semantic=True,
    ),
    retrieval_config=RetrievalConfig(
        dense_per_dimension=80,
        lexical_top_k=120,
        fusion_top_k=120,
        rerank_top_k=30,
        use_diversity=True,
        auto_expand_context=True,
        context_token_budget=5000,
    ),
    persistence_path="knowledge.db",
)

engine.ingest(
    "# PostgreSQL\nPostgreSQL is relational.\n\n## Errors\nERR_DB-7312 means ...",
    source_id="postgres-guide",
    metadata={"tenant_id": "acme", "visibility": "public", "source_trust": 1.0},
)

result = engine.search("ERR_DB-7312", top_k=5, tenant_id="acme")
for hit in result.hits:
    print(hit.score, hit.document.id, hit.reasons)

print(result.context_documents)
engine.close()
```

If no retrieval stage produces evidence, the result is empty; the engine never manufactures arbitrary fallback candidates.

## Neural embeddings

`HashEmbeddingProvider` is a deterministic test/offline fallback. Production semantic retrieval should normally use a neural provider.

```python
from semantic_chunk_search import SentenceTransformerEmbeddingProvider, SemanticChunkSearch

embedder = SentenceTransformerEmbeddingProvider("intfloat/multilingual-e5-base")
engine = SemanticChunkSearch(embedder=embedder)
```

The facade clamps chunk size to the embedding model's input budget when the provider exposes `max_input_tokens`.

## Sparse / SPLADE

```python
from semantic_chunk_search import SentenceTransformerSparseEncoder, SemanticChunkSearch

sparse = SentenceTransformerSparseEncoder("naver/splade-cocondenser-ensembledistil")
engine = SemanticChunkSearch(sparse_encoder=sparse)
```

## Cross encoder and late interaction

```python
from semantic_chunk_search import CrossEncoderReranker, SemanticChunkSearch

engine = SemanticChunkSearch(
    reranker=CrossEncoderReranker(),
)
```

`CrossEncoderReranker` has an explicit activation contract (`model`, `sigmoid`, `identity`) so scores are not accidentally activated twice.

For real token-matrix late interaction, use `BaseLateInteractionEncoder` / `CallableLateInteractionEncoder` with `MultiVectorLateInteractionReranker`. `TokenMaxSimReranker` remains a lightweight approximation.

## Persistent, idempotent ingestion

```python
engine = SemanticChunkSearch(persistence_path="ssc.db")
engine.ingest(text, "manual", source_version="7", metadata={"visibility": "private"})
engine.delete_source("manual")
engine.close()
```

Re-ingesting unchanged text still applies changed ACL/metadata. SQLite uses WAL and atomic source replacement. The facade attempts snapshot rollback if a source update/delete fails partway through.

## Files and provenance

```python
engine.ingest_document("guide.pdf", source_id="guide")
engine.ingest_document("manual.docx", source_id="manual")
engine.ingest_document("page.html", source_id="web")
```

- PDF uses `pypdf`; an OCR provider can be injected for textless pages.
- DOCX preserves headings and tables.
- HTML excludes scripts/styles and hidden DOM content.
- Page/layout metadata is retained when supplied by the loader.
- Search documents retain normalized and original text offsets where available.

Custom page ingestion is also available:

```python
engine.ingest_pages([page_1_text, page_2_text], source_id="annual-report.pdf")
```

## ACL and multi-tenancy

```python
engine.ingest(
    secret_text,
    "payroll",
    metadata={
        "tenant_id": "acme",
        "visibility": "private",
        "allowed_groups": ["finance"],
    },
)

result = engine.search(
    "salary bands",
    tenant_id="acme",
    groups=["finance"],
)
```

ACL/tenant conditions are pushed into candidate retrieval. `source_trust` is applied after optional rankers so a reranker cannot erase a hard policy multiplier.

## Context assembly

Set `auto_expand_context=True`, or use `ContextAssembler` directly. The assembler inserts the retrieved hit first, then optional parent/neighbor context under a token budget. Exact identifier queries remain narrow by default.

## Local ANN

```python
engine = SemanticChunkSearch(embedder=embedder, use_ann=True)
```

This uses `HNSWSemanticIndex` when `hnswlib` is installed. It maintains ANN indexes for all five dense representations and uses HNSW label filtering for allowed candidate sets. `MatryoshkaHNSWSemanticIndex` supports coarse-dimension candidate generation plus full-vector rescoring for MRL-compatible embeddings.

## Qdrant backend

```python
from qdrant_client import QdrantClient
from semantic_chunk_search import QdrantSemanticIndex, DocumentVectorizer, SemanticChunkSearch

client = QdrantClient(url="http://localhost:6333")
vectorizer = DocumentVectorizer(embedder)
index = QdrantSemanticIndex(
    client,
    "knowledge",
    vectorizer,
    dimensions=embedder.dimensions,
    sparse_encoder=sparse,      # optional
    create_collection=True,
    create_payload_indexes=True,
)
engine = SemanticChunkSearch(embedder=embedder, index=index)
```

The backend stores named vectors for `raw_text`, `summary`, `keywords`, `titles`, `references`, plus an optional named sparse vector. It uses source-scoped operations instead of loading the whole corpus into Python memory.

`CallableSemanticIndex` remains available for another database/search engine.

## Multimodal candidates

`MultimodalPageRetriever` supports one image vector per page. `MultiVectorPageRetriever` supports patch/token vectors and MaxSim. A visual retriever can be passed to `SemanticChunkSearch`; visual candidates then participate in the same fusion/ranking pipeline and are still checked by document ACL.

## Feedback and ranking

`FeedbackStore` can be persisted with SQLite. Built-in lightweight ranking options include `LinearLTRReranker` and `PairwiseLinearLTRReranker`; more sophisticated LambdaMART/listwise models can implement the same reranker contract.

## Evaluation

```python
from semantic_chunk_search import EvaluationCase, evaluate

cases = [
    EvaluationCase("postgres replication", relevant_source_ids={"postgres-guide"}),
]
report = evaluate(engine, cases, k=5)
print(report.recall_at_k, report.candidate_recall, report.ndcg_at_k)
```

Source-level relevance is deduplicated before NDCG/Recall. Source-level gold labels are recommended when comparing chunking strategies because chunk boundaries naturally change chunk IDs.

## Production boundary

The bundled components have distinct roles:

- `InMemorySemanticIndex`: correctness/local backend; exact scan, not a million-document serving engine.
- `HNSWSemanticIndex`: local ANN backend; still keeps documents/metadata in process.
- SQLite: durable local snapshot/restart store, not a distributed vector database.
- `QdrantSemanticIndex` / custom `BaseSemanticIndex`: preferred serving path for large corpora.

The security scanner is defense-in-depth and does not replace permission-aware retrieval, source governance or application-level authorization.
