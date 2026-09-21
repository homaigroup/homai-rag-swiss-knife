from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol

_IDENTIFIER_RE = re.compile(r"(?:https?://\S+|\b[A-Z]{2,}[\-_]?\d{2,}\b|\b\w+[\-_]\d{3,}\b)", re.I)
_COMPARE_RE = re.compile(r"\b(vs\.?|versus|compare|difference|تفاوت|مقایسه)\b", re.I)
_GLOBAL_RE = re.compile(r"\b(across|overall|all reports|trend|summary of all|در همه|کل اسناد|روند)\b", re.I)
_CODE_RE = re.compile(r"(?:\b(def|class|function|method|stack trace|exception|api)\b|[A-Za-z_][A-Za-z0-9_]+\(\))", re.I)
_RELATION_RE = re.compile(r"\b(who|related|relationship|worked with|manager|owner|چه کسی|مرتبط|رابطه|مدیر)\b", re.I)

@dataclass(slots=True)
class RetrievalPolicy:
    name: str = "balanced"
    dense_weight: float = 1.0
    lexical_weight: float = 1.0
    sparse_weight: float = 0.7
    exact_weight: float = 1.5
    use_multi_query: bool = False
    use_graph: bool = False

class QueryPolicyClassifier(Protocol):
    def classify(self, query: str) -> tuple[RetrievalPolicy, float] | RetrievalPolicy: ...

class QueryRouter:
    """Deterministic fallback router with an optional learned/model classifier.

    The classifier is only trusted above ``min_confidence``. This keeps routing
    deterministic and dependency-free by default, while allowing a trained small
    model/LLM to override ambiguous cases in production.
    """
    def __init__(self,classifier:QueryPolicyClassifier|None=None,min_confidence:float=0.75)->None:
        if not 0<=min_confidence<=1:raise ValueError("min_confidence must be between 0 and 1")
        self.classifier=classifier;self.min_confidence=min_confidence
    def _heuristic(self,query:str)->RetrievalPolicy:
        if _IDENTIFIER_RE.search(query):return RetrievalPolicy("exact-heavy",dense_weight=0.5,lexical_weight=1.5,sparse_weight=0.3,exact_weight=2.0)
        if _CODE_RE.search(query):return RetrievalPolicy("code",dense_weight=0.8,lexical_weight=1.4,sparse_weight=0.6,exact_weight=1.7)
        if _COMPARE_RE.search(query):return RetrievalPolicy("comparative",dense_weight=1.2,lexical_weight=0.8,sparse_weight=0.9,use_multi_query=True)
        if _GLOBAL_RE.search(query):return RetrievalPolicy("global",dense_weight=1.2,lexical_weight=0.5,sparse_weight=1.0,use_multi_query=True,use_graph=True)
        if _RELATION_RE.search(query):return RetrievalPolicy("entity-relational",dense_weight=1.0,lexical_weight=0.8,sparse_weight=0.7,use_multi_query=True,use_graph=True)
        if len(query.split())<=3:return RetrievalPolicy("short",dense_weight=0.8,lexical_weight=1.2,sparse_weight=0.7)
        return RetrievalPolicy()
    def route(self,query:str)->RetrievalPolicy:
        if self.classifier is not None:
            try:
                result=self.classifier.classify(query)
                if isinstance(result,tuple):policy,confidence=result
                else:policy,confidence=result,1.0
                if isinstance(policy,RetrievalPolicy) and float(confidence)>=self.min_confidence:return policy
            except Exception:
                pass
        return self._heuristic(query)

class QueryDecomposer(Protocol):
    def decompose(self, query: str) -> list[str]: ...

class HeuristicQueryDecomposer:
    def decompose(self, query: str) -> list[str]:
        parts=[p.strip() for p in re.split(r"\b(?:and|then|also|و|همچنین|سپس)\b|[;؛]",query,flags=re.I) if p.strip()]
        if len(parts)<=1:return [query]
        return list(dict.fromkeys([query,*parts]))[:5]

class LLMQueryDecomposer:
    def __init__(self,llm)->None:self.llm=llm
    def decompose(self,query:str)->list[str]:
        try:
            raw=self.llm.generate("Decompose this search question into at most 4 independent retrieval subqueries, one per line, no numbering: "+query)
            parts=[x.strip(" -\t") for x in raw.splitlines() if x.strip()]
            return list(dict.fromkeys([query,*parts]))[:5]
        except Exception:return [query]

def heuristic_decompose(query:str)->list[str]:return HeuristicQueryDecomposer().decompose(query)
