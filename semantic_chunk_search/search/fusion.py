from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    """Return normalized RRF scores in [0, 1]."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights length must match rankings length")

    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + float(weight) / (k + rank)
    if not scores:
        return {}
    maximum = max(scores.values())
    if maximum <= 0.0:
        return scores
    return {doc_id: score / maximum for doc_id, score in scores.items()}
