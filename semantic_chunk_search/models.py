from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import UUID


class ChunkType(str, Enum):
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BASED = "token_based"
    MARKDOWN = "markdown"
    HIERARCHICAL = "hierarchical"
    CODE = "code"
    TABLE = "table"
    COMPOSITE = "composite"


@dataclass(slots=True)
class Chunk:
    raw_text: str
    source_id: str | UUID
    chunk_index: int
    chunk_type: ChunkType
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = ""
    parent_id: str | None = None
    start_char: int | None = None
    end_char: int | None = None


@dataclass(slots=True)
class ChunkingConfig:
    strategy: str = "auto"
    max_chunk_tokens: int = 1024
    min_chunk_tokens: int = 100
    overlap_tokens: int = 50
    semantic_threshold: float = 0.50
    adaptive_semantic: bool = True
    semantic_break_percentile: float = 20.0
    preserve_structure: bool = True
    tokenizer_name: str | None = None

    def __post_init__(self) -> None:
        if self.max_chunk_tokens < 1: raise ValueError("max_chunk_tokens must be >= 1")
        if self.min_chunk_tokens < 1: raise ValueError("min_chunk_tokens must be >= 1")
        if self.min_chunk_tokens > self.max_chunk_tokens: raise ValueError("min_chunk_tokens cannot exceed max_chunk_tokens")
        if self.overlap_tokens < 0: raise ValueError("overlap_tokens must be >= 0")
        if self.overlap_tokens >= self.max_chunk_tokens: raise ValueError("overlap_tokens must be smaller than max_chunk_tokens")
        if not 0.0 <= self.semantic_threshold <= 1.0: raise ValueError("semantic_threshold must be between 0 and 1")
        if not 0.0 <= self.semantic_break_percentile <= 100.0: raise ValueError("semantic_break_percentile must be between 0 and 100")


@dataclass(slots=True)
class MultiVector:
    raw_text: list[float] = field(default_factory=list)
    summary: list[float] = field(default_factory=list)
    keywords: list[float] = field(default_factory=list)
    titles: list[float] = field(default_factory=list)
    references: list[float] = field(default_factory=list)

    def dimensions(self) -> dict[str, list[float]]:
        return {"raw_text": self.raw_text, "summary": self.summary, "keywords": self.keywords, "titles": self.titles, "references": self.references}


@dataclass(slots=True)
class SearchDocument:
    id: str
    text: str
    source_id: str | UUID | None = None
    chunk_index: int | None = None
    summary: str = ""
    keywords: list[str] = field(default_factory=list)
    titles: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    vectors: MultiVector | None = None
    context_text: str = ""
    parent_id: str | None = None
    section_path: list[str] = field(default_factory=list)
    previous_id: str | None = None
    next_id: str | None = None
    source_version: str = "1"
    content_hash: str = ""
    embedding_model: str = ""
    embedding_dimensions: int = 0
    embedding_version: str = "4"
    embedding_signature: str = ""
    vector_schema_version: str = "4"
    chunking_version: str = "4"
    chunking_signature: str = ""
    enrichment_version: str = "4"
    retrieval_signature: str = ""
    created_at: str = ""
    page_start: int | None = None
    page_end: int | None = None
    start_char: int | None = None
    end_char: int | None = None
    normalized_start_char: int | None = None
    normalized_end_char: int | None = None


@dataclass(slots=True)
class ParentDocument:
    id: str
    source_id: str | UUID
    text: str
    section_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    child_ids: list[str] = field(default_factory=list)
    token_count: int = 0


@dataclass(slots=True)
class RetrievalWeights:
    raw_text: float = 0.30
    summary: float = 0.25
    keywords: float = 0.15
    titles: float = 0.20
    references: float = 0.10

    def __post_init__(self) -> None:
        values=[self.raw_text,self.summary,self.keywords,self.titles,self.references]
        if any(v < 0 for v in values): raise ValueError("retrieval weights cannot be negative")
        total=sum(values)
        if total <= 0: raise ValueError("retrieval weights must sum to a positive value")
        self.raw_text/=total; self.summary/=total; self.keywords/=total; self.titles/=total; self.references/=total

    def as_dict(self) -> dict[str,float]:
        return {"raw_text":self.raw_text,"summary":self.summary,"keywords":self.keywords,"titles":self.titles,"references":self.references}


@dataclass(slots=True)
class RetrievalConfig:
    hybrid: bool = True
    candidate_multiplier: int = 10
    min_candidates: int = 50
    dense_per_dimension: int = 80
    lexical_top_k: int = 120
    sparse_top_k: int = 120
    exact_top_k: int = 50
    visual_top_k: int = 80
    visual_weight: float = 0.8
    fusion_top_k: int = 120
    late_interaction_top_k: int = 60
    rerank_top_k: int = 30
    rrf_k: int = 60
    semantic_blend: float = 0.65
    use_multi_representation: bool = True
    use_query_router: bool = True
    use_multi_query: bool = True
    use_diversity: bool = True
    mmr_lambda: float = 0.78
    max_per_source: int | None = 4
    auto_expand_context: bool = False
    context_token_budget: int = 4000
    context_neighbors: int = 1
    feedback_weight: float = 0.05
    tenant_mode: str = "isolated"  # isolated | shared_public | global
    fail_open_optional_stages: bool = True

    def __post_init__(self) -> None:
        if self.candidate_multiplier < 1 or self.min_candidates < 1: raise ValueError("candidate limits must be >= 1")
        for name in ("dense_per_dimension","lexical_top_k","sparse_top_k","exact_top_k","visual_top_k","fusion_top_k","late_interaction_top_k","rerank_top_k"):
            if getattr(self,name) < 1: raise ValueError(f"{name} must be >= 1")
        if self.visual_weight < 0: raise ValueError("visual_weight must be >= 0")
        if self.rrf_k < 1: raise ValueError("rrf_k must be >= 1")
        if not 0.0 <= self.semantic_blend <= 1.0: raise ValueError("semantic_blend must be between 0 and 1")
        if not 0.0 <= self.mmr_lambda <= 1.0: raise ValueError("mmr_lambda must be between 0 and 1")
        if self.context_token_budget < 1: raise ValueError("context_token_budget must be >= 1")
        if not 0.0 <= self.feedback_weight <= 0.5: raise ValueError("feedback_weight must be between 0 and 0.5")
        if self.tenant_mode not in {"isolated","shared_public","global"}: raise ValueError("tenant_mode must be isolated, shared_public, or global")


@dataclass(slots=True)
class DimensionScores:
    raw_text: float = 0.0
    summary: float = 0.0
    keywords: float = 0.0
    titles: float = 0.0
    references: float = 0.0


@dataclass(slots=True)
class SearchHit:
    document: SearchDocument
    score: float
    dimension_scores: DimensionScores
    dense_score: float = 0.0
    lexical_score: float = 0.0
    sparse_score: float = 0.0
    exact_score: float = 0.0
    graph_score: float = 0.0
    visual_score: float = 0.0
    fusion_score: float = 0.0
    semantic_score: float = 0.0
    late_interaction_score: float | None = None
    rerank_score: float | None = None
    relevance_score: float = 0.0
    feedback_score: float = 0.0
    policy_multiplier: float = 1.0
    calibrated_score: float | None = None
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SearchResult:
    query: str
    hits: list[SearchHit]
    total_found: int
    retrieval_mode: str = "hybrid"
    diagnostics: dict[str, Any] = field(default_factory=dict)
    context_documents: list[SearchDocument] = field(default_factory=list)
