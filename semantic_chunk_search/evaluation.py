from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Protocol


class Searcher(Protocol):
    def search(self, query: str, top_k: int = 10, **kwargs): ...


@dataclass(slots=True)
class EvaluationCase:
    query: str
    relevant_ids: set[str] = field(default_factory=set)
    relevant_source_ids: set[str] = field(default_factory=set)
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class QueryMetrics:
    query: str
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    candidate_recall: float
    source_diversity: float
    latency_ms: float
    retrieved_ids: list[str]


@dataclass(slots=True)
class EvaluationReport:
    k: int
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    candidate_recall: float
    source_diversity: float
    mean_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    cases: list[QueryMetrics]


def precision_at_k(retrieved:list[str],relevant:set[str],k:int)->float:
    return len(set(retrieved[:k])&relevant)/k if k>0 else 0.0

def recall_at_k(retrieved:list[str],relevant:set[str],k:int)->float:
    return 1.0 if not relevant else len(set(retrieved[:k])&relevant)/len(relevant)

def reciprocal_rank(retrieved:list[str],relevant:set[str])->float:
    for rank,doc_id in enumerate(retrieved,1):
        if doc_id in relevant:return 1.0/rank
    return 0.0

def ndcg_at_k(retrieved:list[str],relevant:set[str],k:int)->float:
    dcg=sum(1.0/math.log2(rank+1.0) for rank,doc_id in enumerate(retrieved[:k],1) if doc_id in relevant)
    ideal=min(len(relevant),k); idcg=sum(1.0/math.log2(rank+1.0) for rank in range(1,ideal+1))
    return dcg/idcg if idcg else 1.0


def _percentile(values:list[float],p:float)->float:
    if not values:return 0.0
    ordered=sorted(values); idx=max(0,min(len(ordered)-1,math.ceil(p*len(ordered))-1)); return ordered[idx]


def evaluate(searcher:Searcher,cases:list[EvaluationCase],k:int=10,**search_kwargs)->EvaluationReport:
    if k<1:raise ValueError("k must be >= 1")
    rows=[]
    for case in cases:
        start=time.perf_counter(); result=searcher.search(case.query,top_k=k,**search_kwargs); latency=(time.perf_counter()-start)*1000.0
        ids=[h.document.id for h in result.hits]; candidate_ids=list(result.diagnostics.get("candidate_ids",ids)); sources=[str(h.document.source_id) for h in result.hits]
        if case.relevant_ids:
            eval_retrieved=ids; relevant=case.relevant_ids; candidate_eval=candidate_ids
        else:
            # Source-level evaluation treats repeated chunks from the same source as one ranked source.
            eval_retrieved=list(dict.fromkeys(sources)); relevant=case.relevant_source_ids
            index=getattr(searcher,"index",None)
            if index is None and hasattr(searcher,"search_engine"): index=getattr(searcher.search_engine,"index",None)
            candidate_eval=[]
            for cid in candidate_ids:
                doc=index.get(cid) if index is not None else None
                candidate_eval.append(str(doc.source_id) if doc is not None else cid)
            candidate_eval=list(dict.fromkeys(candidate_eval))
        rows.append(QueryMetrics(case.query,precision_at_k(eval_retrieved,relevant,k),recall_at_k(eval_retrieved,relevant,k),reciprocal_rank(eval_retrieved,relevant),ndcg_at_k(eval_retrieved,relevant,k),recall_at_k(candidate_eval,relevant,len(candidate_eval) or 1),len(set(sources))/max(len(sources),1),latency,ids))
    if not rows:return EvaluationReport(k,0,0,0,0,0,0,0,0,0,[])
    lat=[r.latency_ms for r in rows]
    return EvaluationReport(k,statistics.mean(r.precision_at_k for r in rows),statistics.mean(r.recall_at_k for r in rows),statistics.mean(r.reciprocal_rank for r in rows),statistics.mean(r.ndcg_at_k for r in rows),statistics.mean(r.candidate_recall for r in rows),statistics.mean(r.source_diversity for r in rows),statistics.mean(lat),_percentile(lat,.95),_percentile(lat,.99),rows)


def hard_negative_case(query:str,relevant_id:str,*confusing_ids:str)->EvaluationCase:
    return EvaluationCase(query=query,relevant_ids={relevant_id},metadata={"hard_negatives":list(confusing_ids)})


def boundary_f1(predicted_boundaries:list[int],gold_boundaries:list[int],tolerance:int=0)->tuple[float,float,float]:
    """Precision/recall/F1 for chunk boundary positions with optional character/token tolerance."""
    pred=list(predicted_boundaries); gold=list(gold_boundaries); matched=set(); tp=0
    for p in pred:
        candidates=[(abs(p-g),i) for i,g in enumerate(gold) if i not in matched and abs(p-g)<=tolerance]
        if candidates:
            _,idx=min(candidates); matched.add(idx); tp+=1
    precision=tp/len(pred) if pred else (1.0 if not gold else 0.0)
    recall=tp/len(gold) if gold else 1.0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    return precision,recall,f1


def citation_coverage(cited_source_ids:set[str],required_source_ids:set[str])->float:
    if not required_source_ids:return 1.0
    return len(cited_source_ids & required_source_ids)/len(required_source_ids)
