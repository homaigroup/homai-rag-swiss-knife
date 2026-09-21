from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .evaluation import EvaluationCase, EvaluationReport, evaluate
from .models import ChunkingConfig


@dataclass(slots=True)
class OptimizationResult:
    config: ChunkingConfig
    report: EvaluationReport
    objective: float


class ChunkingPolicyOptimizer:
    """Benchmark chunking configurations against the user's gold queries instead of guessing one global size."""
    def __init__(self, engine_factory: Callable[[ChunkingConfig], object]) -> None:
        self.engine_factory=engine_factory

    def optimize(self,corpus:list[tuple[str,str,dict|None]],cases:list[EvaluationCase],configs:list[ChunkingConfig],k:int=10)->list[OptimizationResult]:
        results=[]
        for config in configs:
            engine=self.engine_factory(config)
            for text,source_id,metadata in corpus: engine.ingest(text,source_id,metadata=metadata)
            report=evaluate(engine,cases,k=k)
            objective=.45*report.recall_at_k+.25*report.candidate_recall+.2*report.ndcg_at_k+.1*report.mrr
            results.append(OptimizationResult(config,report,objective))
            close=getattr(engine,"close",None)
            if close: close()
        return sorted(results,key=lambda r:r.objective,reverse=True)
