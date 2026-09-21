from semantic_chunk_search import (
    ChunkingConfig,
    ChunkingFactory,
    HashEmbeddingProvider,
    SearchDocument,
    SemanticChunkSearch,
)
from semantic_chunk_search.search import DocumentVectorizer, InMemorySemanticIndex, QueryEnricher, SemanticSearchEngine


def test_all_chunking_strategies_run():
    embedder = HashEmbeddingProvider(128)
    cfg = ChunkingConfig(max_chunk_tokens=30, min_chunk_tokens=5, overlap_tokens=5)
    factory = ChunkingFactory(cfg, embedder)
    samples = {
        "recursive": "One paragraph about systems. " * 20,
        "semantic": "Cats are domestic animals. They often live with people. Databases store structured information. SQL queries tables.",
        "sliding_window": "word " * 100,
        "token_based": "token " * 100,
        "markdown": "# Intro\nHello world.\n\n## Data\nDatabase content here.",
        "code": "def hello():\n    return 'world'\n\nclass A:\n    pass",
    }
    for strategy, text in samples.items():
        chunks = factory.chunk(text, "source-1", strategy=strategy)
        assert chunks
        assert all(c.source_id == "source-1" for c in chunks)


def test_semantic_search_end_to_end():
    embedder = HashEmbeddingProvider(256)
    index = InMemorySemanticIndex(DocumentVectorizer(embedder))
    index.add_many([
        SearchDocument(id="db", text="PostgreSQL is a relational database. SQL queries relational tables.", keywords=["database", "sql"]),
        SearchDocument(id="cat", text="Cats are small domestic animals that often live with humans.", keywords=["cat", "animal"]),
    ])
    engine = SemanticSearchEngine(index, QueryEnricher(embedder))
    result = engine.search("SQL relational database", top_k=1)
    assert result.hits
    assert result.hits[0].document.id == "db"


def test_facade_ingest_and_search():
    engine = SemanticChunkSearch(ChunkingConfig(strategy="semantic", max_chunk_tokens=35, min_chunk_tokens=4, overlap_tokens=2))
    docs = engine.ingest(
        "Python is a programming language. Developers write software with Python. PostgreSQL stores relational data. SQL retrieves database rows.",
        "doc-x",
    )
    assert docs
    result = engine.search("database SQL", top_k=2)
    assert result.hits
