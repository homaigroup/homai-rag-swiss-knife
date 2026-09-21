from semantic_chunk_search import ChunkingConfig, RetrievalConfig, SemanticChunkSearch

engine = SemanticChunkSearch(
    ChunkingConfig(strategy="auto", max_chunk_tokens=80, min_chunk_tokens=8, overlap_tokens=8),
    retrieval_config=RetrievalConfig(auto_expand_context=True, context_token_budget=300),
)

engine.ingest(
    """# API Guide
This guide explains authentication.

```python
def login(token):
    return token is not None
```

## Database
PostgreSQL stores relational rows. ERR_DB-7312 is raised when the database connection is unavailable.

| Code | Meaning |
|---|---|
| ERR_DB-7312 | Database unavailable |
""",
    "api-guide",
    metadata={"tenant_id": "demo", "visibility": "public", "source_trust": 1.0},
)

result = engine.search("ERR_DB-7312", top_k=3, tenant_id="demo")
for hit in result.hits:
    print(round(hit.score, 4), hit.document.metadata.get("block_type"), hit.reasons)

print("trace ms:", result.diagnostics["trace"]["total_ms"])
