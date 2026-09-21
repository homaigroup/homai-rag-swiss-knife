from __future__ import annotations

from ..embedding import cosine_similarity
from ..models import DimensionScores, MultiVector, RetrievalWeights


def _positive_cosine(a: list[float], b: list[float]) -> float:
    """Map unrelated/negative cosine to 0 instead of giving it a 0.5 baseline."""
    if not a or not b:
        return 0.0
    return max(0.0, cosine_similarity(a, b))


def dimension_scores(query: MultiVector, document: MultiVector) -> DimensionScores:
    return DimensionScores(
        raw_text=_positive_cosine(query.raw_text, document.raw_text),
        summary=_positive_cosine(query.summary, document.summary),
        keywords=_positive_cosine(query.keywords, document.keywords),
        titles=_positive_cosine(query.titles, document.titles),
        references=_positive_cosine(query.references, document.references),
    )


def weighted_score(
    scores: DimensionScores,
    weights: RetrievalWeights,
    query: MultiVector | None = None,
    document: MultiVector | None = None,
) -> float:
    """Weighted multi-vector score with renormalization over dimensions that exist.

    Older code can still call ``weighted_score(scores, weights)``. The search engine
    passes query/document vectors so missing dimensions do not unfairly penalize a hit.
    """
    items = [
        ("raw_text", weights.raw_text),
        ("summary", weights.summary),
        ("keywords", weights.keywords),
        ("titles", weights.titles),
        ("references", weights.references),
    ]
    if query is None or document is None:
        return sum(getattr(scores, name) * weight for name, weight in items)

    active: list[tuple[str, float]] = []
    for name, weight in items:
        if getattr(query, name) and getattr(document, name):
            active.append((name, weight))
    if not active:
        return 0.0
    total_weight = sum(weight for _, weight in active)
    return sum(getattr(scores, name) * weight for name, weight in active) / total_weight
