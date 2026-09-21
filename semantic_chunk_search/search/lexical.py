from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable

from ..normalization import DEFAULT_NORMALIZER, TextNormalizer

_TOKEN_RE = re.compile(r"[\w\-]+", re.UNICODE)
_REFERENCE_RE = re.compile(r"(?:https?://\S+|\b[A-Z]{2,}[\-_]?\d{2,}\b|\b\w+[\-_]\d{3,}\b)", re.I)


def lexical_tokens(text: str, normalizer: TextNormalizer | None = None) -> list[str]:
    normalized = (normalizer or DEFAULT_NORMALIZER).normalize_for_search(text)
    return [t for t in _TOKEN_RE.findall(normalized) if t]


class BM25Index:
    """Dependency-free inverted BM25 index with filterable candidate scoring."""
    def __init__(self, k1: float = 1.5, b: float = 0.75, normalizer: TextNormalizer | None = None) -> None:
        self.k1, self.b = k1, b
        self.normalizer = normalizer or DEFAULT_NORMALIZER
        self._tf: dict[str, Counter[str]] = {}
        self._lengths: dict[str, int] = {}
        self._postings: dict[str, dict[str, int]] = defaultdict(dict)

    def add(self, document_id: str, text: str) -> None:
        self.remove(document_id)
        counts = Counter(lexical_tokens(text, self.normalizer))
        self._tf[document_id] = counts; self._lengths[document_id] = sum(counts.values())
        for term, freq in counts.items(): self._postings[term][document_id] = freq

    def remove(self, document_id: str) -> None:
        old = self._tf.pop(document_id, None); self._lengths.pop(document_id, None)
        if not old: return
        for term in old:
            bucket = self._postings.get(term)
            if bucket is not None:
                bucket.pop(document_id, None)
                if not bucket: self._postings.pop(term, None)

    def clear(self) -> None:
        self._tf.clear(); self._lengths.clear(); self._postings.clear()

    def search(self, query: str, top_k: int, allowed_ids: Iterable[str] | None = None) -> list[tuple[str, float]]:
        query_terms = lexical_tokens(query, self.normalizer)
        if not query_terms or not self._tf: return []
        allowed = set(allowed_ids) if allowed_ids is not None else None
        n_docs = len(self._tf); avgdl = sum(self._lengths.values()) / max(n_docs, 1)
        candidates: set[str] = set()
        for term in query_terms: candidates.update(self._postings.get(term, {}))
        if allowed is not None: candidates &= allowed
        scores: dict[str, float] = defaultdict(float)
        for term in query_terms:
            postings = self._postings.get(term, {}); df = len(postings)
            if not df: continue
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for doc_id in candidates:
                freq = postings.get(doc_id, 0)
                if not freq: continue
                dl = self._lengths.get(doc_id, 0)
                denom = freq + self.k1 * (1.0 - self.b + self.b * dl / max(avgdl, 1e-9))
                scores[doc_id] += idf * (freq * (self.k1 + 1.0)) / denom
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


class ExactTermIndex:
    def __init__(self, normalizer: TextNormalizer | None = None) -> None:
        self.normalizer = normalizer or DEFAULT_NORMALIZER
        self._terms: dict[str, set[str]] = defaultdict(set)
        self._by_doc: dict[str, set[str]] = {}

    def add(self, document_id: str, text: str, extra_terms: Iterable[str] = ()) -> None:
        self.remove(document_id)
        refs = set(_REFERENCE_RE.findall(text)) | {str(t) for t in extra_terms if str(t).strip()}
        terms = {self.normalizer.normalize_for_search(t) for t in refs}
        self._by_doc[document_id] = terms
        for term in terms: self._terms[term].add(document_id)

    def remove(self, document_id: str) -> None:
        for term in self._by_doc.pop(document_id, set()):
            self._terms[term].discard(document_id)
            if not self._terms[term]: self._terms.pop(term, None)

    def clear(self) -> None:
        self._terms.clear(); self._by_doc.clear()

    def search(self, query: str, top_k: int, allowed_ids: set[str] | None = None) -> list[tuple[str, float]]:
        q = self.normalizer.normalize_for_search(query)
        scores: dict[str, float] = defaultdict(float)
        for term, doc_ids in self._terms.items():
            if term and term in q:
                for doc_id in doc_ids:
                    if allowed_ids is None or doc_id in allowed_ids:
                        scores[doc_id] += 1.0 + min(1.0, len(term) / max(len(q), 1))
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

class BM25FIndex:
    """Field-aware BM25F-style fusion over independent inverted BM25 fields."""
    def __init__(self, field_weights: dict[str,float] | None = None, normalizer: TextNormalizer | None = None) -> None:
        self.field_weights=field_weights or {"body":1.0,"summary":1.15,"keywords":1.35,"titles":1.8,"references":1.6}
        self.normalizer=normalizer or DEFAULT_NORMALIZER
        self._fields={name:BM25Index(normalizer=self.normalizer) for name in self.field_weights}
        self._docs:set[str]=set()
    def add(self,document_id:str,fields:dict[str,str])->None:
        self.remove(document_id);self._docs.add(document_id)
        for name,index in self._fields.items():index.add(document_id,str(fields.get(name,"") or ""))
    def remove(self,document_id:str)->None:
        self._docs.discard(document_id)
        for index in self._fields.values():index.remove(document_id)
    def clear(self)->None:
        self._docs.clear()
        for index in self._fields.values():index.clear()
    def search(self,query:str,top_k:int,allowed_ids:Iterable[str]|None=None)->list[tuple[str,float]]:
        allowed=set(allowed_ids) if allowed_ids is not None else None;scores:dict[str,float]=defaultdict(float)
        for name,index in self._fields.items():
            weight=float(self.field_weights.get(name,1.0))
            for doc_id,score in index.search(query,max(top_k*3,top_k),allowed):scores[doc_id]+=weight*score
        return sorted(scores.items(),key=lambda x:x[1],reverse=True)[:top_k]
