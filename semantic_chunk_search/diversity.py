from __future__ import annotations

from .embedding import cosine_similarity
from .models import SearchHit


def mmr_select(hits: list[SearchHit], top_k: int, lambda_mult: float = 0.75, max_per_source: int | None = None) -> list[SearchHit]:
    if not hits or top_k <= 0:
        return []
    lambda_mult = max(0.0, min(1.0, lambda_mult))
    selected: list[SearchHit] = []
    remaining = list(hits)
    source_counts: dict[str, int] = {}
    while remaining and len(selected) < top_k:
        best = None
        best_score = float("-inf")
        for hit in remaining:
            source = str(hit.document.source_id or "")
            if max_per_source is not None and source_counts.get(source, 0) >= max_per_source:
                continue
            novelty_penalty = 0.0
            if selected and hit.document.vectors and hit.document.vectors.raw_text:
                similarities=[max(0.0, cosine_similarity(hit.document.vectors.raw_text, s.document.vectors.raw_text)) for s in selected if s.document.vectors and s.document.vectors.raw_text]
                novelty_penalty=max(similarities) if similarities else 0.0
            score = lambda_mult * hit.score - (1.0 - lambda_mult) * novelty_penalty
            if score > best_score:
                best_score = score
                best = hit
        if best is None:
            break
        remaining.remove(best)
        selected.append(best)
        source = str(best.document.source_id or "")
        source_counts[source] = source_counts.get(source, 0) + 1
    return selected
