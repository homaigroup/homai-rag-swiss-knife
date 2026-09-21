from semantic_chunk_search import (
    ChunkingConfig,
    ChunkingFactory,
    EvaluationCase,
    HashEmbeddingProvider,
    RetrievalConfig,
    SearchDocument,
    SemanticChunkSearch,
    evaluate,
)
from semantic_chunk_search.search import DocumentVectorizer, InMemorySemanticIndex, QueryEnricher, SemanticSearchEngine


def test_unrelated_cosine_no_half_baseline():
    embedder = HashEmbeddingProvider(512)
    index = InMemorySemanticIndex(DocumentVectorizer(embedder))
    index.add(SearchDocument(id="cats", text="cats dogs pets animals"))
    engine = SemanticSearchEngine(index, QueryEnricher(embedder), retrieval_config=RetrievalConfig(hybrid=False))
    result = engine.search("postgresql database sql", top_k=1, hybrid=False)
    # No synthetic +0.5 offset: unrelated text should not receive a large semantic score.
    assert not result.hits or result.hits[0].semantic_score < 0.35


def test_document_multivectors_are_distinct_views():
    embedder = HashEmbeddingProvider(256)
    vectorizer = DocumentVectorizer(embedder)
    doc = SearchDocument(
        id="d1",
        text="PostgreSQL stores relational rows. SQL queries tables. SQL indexes accelerate lookup.",
        section_path=["Database Guide", "Indexes"],
    )
    vectors = vectorizer.vectorize(doc)
    assert vectors.raw_text
    assert vectors.summary
    assert vectors.keywords
    assert vectors.titles
    assert vectors.raw_text != vectors.keywords
    assert vectors.raw_text != vectors.titles


def test_hybrid_bm25_recovers_exact_identifier():
    embedder = HashEmbeddingProvider(256)
    index = InMemorySemanticIndex(DocumentVectorizer(embedder))
    index.add_many([
        SearchDocument(id="target", text="Incident ERR_X9-7312 occurred during checkout processing."),
        SearchDocument(id="other", text="General checkout troubleshooting and payment documentation."),
    ])
    engine = SemanticSearchEngine(index, QueryEnricher(embedder))
    result = engine.search("ERR_X9-7312", top_k=1)
    assert result.hits[0].document.id == "target"
    assert result.hits[0].lexical_score > 0
    assert result.retrieval_mode == "hybrid"


def test_hierarchical_chunking_preserves_section_path_and_parent():
    factory = ChunkingFactory(ChunkingConfig(max_chunk_tokens=40, min_chunk_tokens=3, overlap_tokens=2))
    chunks = factory.chunk(
        "# Product Manual\nIntro text.\n\n## Installation\nInstall package and configure environment.\n\n### Linux\nRun the Linux command.",
        "manual-1",
        strategy="hierarchical",
    )
    assert chunks
    linux = [c for c in chunks if "Linux" in c.metadata.get("section_path", [])]
    assert linux
    assert linux[0].metadata["parent_id"].startswith("manual-1:section:")
    assert linux[0].metadata["section_path"] == ["Product Manual", "Installation", "Linux"]


def test_facade_context_window_and_hybrid_search():
    engine = SemanticChunkSearch(
        ChunkingConfig(strategy="hierarchical", max_chunk_tokens=25, min_chunk_tokens=3, overlap_tokens=2)
    )
    docs = engine.ingest(
        "# Guide\nIntro.\n\n## Database\nPostgreSQL is relational. SQL retrieves rows. Indexes speed queries.\n\n## Animals\nCats are pets.",
        "guide",
    )
    result = engine.search("SQL PostgreSQL", top_k=1)
    assert result.hits
    window = engine.context_window(result.hits[0].document.id, neighbors=1)
    assert window
    assert all(d.source_id == "guide" for d in window)
    assert docs


def test_evaluation_suite():
    embedder = HashEmbeddingProvider(256)
    index = InMemorySemanticIndex(DocumentVectorizer(embedder))
    index.add_many([
        SearchDocument(id="db", text="PostgreSQL SQL relational database"),
        SearchDocument(id="cat", text="Cats domestic pets animals"),
    ])
    engine = SemanticSearchEngine(index, QueryEnricher(embedder))
    report = evaluate(engine, [EvaluationCase("SQL database", {"db"})], k=1)
    assert report.recall_at_k == 1.0
    assert report.mrr == 1.0
    assert report.ndcg_at_k == 1.0
    assert report.mean_latency_ms >= 0.0
