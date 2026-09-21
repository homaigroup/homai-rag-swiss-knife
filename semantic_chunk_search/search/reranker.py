from __future__ import annotations

import math
from typing import Protocol

from ..models import SearchHit

class Reranker(Protocol):
    def rerank(self,query:str,hits:list[SearchHit],top_k:int)->list[SearchHit]:...

class CrossEncoderReranker:
    """Cross-encoder ordering stage without implicit double activation.

    ``activation='model'`` trusts the model's configured output. Use ``sigmoid`` only
    when the selected model is known to return raw logits.
    """
    def __init__(self,model_name:str="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",blend:float|None=None,activation:str="sigmoid")->None:
        if blend is not None and not 0<=blend<=1:raise ValueError("blend must be between 0 and 1")
        if activation not in {"model","sigmoid","identity"}:raise ValueError("activation must be model, sigmoid, or identity")
        try:from sentence_transformers import CrossEncoder
        except ImportError as exc:raise ImportError("Install sentence-transformers or semantic-chunk-search[reranker]") from exc
        self.model=CrossEncoder(model_name);self.blend=blend;self.activation=activation
    def _score(self,value)->float:
        return float(value)
    def rerank(self,query:str,hits:list[SearchHit],top_k:int)->list[SearchHit]:
        candidates=list(hits)
        if not candidates:return []
        pairs=[(query,h.document.context_text or h.document.text) for h in candidates]
        if self.activation=="sigmoid":
            try:
                from torch.nn import Sigmoid
                raw=self.model.predict(pairs,activation_fn=Sigmoid())
            except (ImportError,TypeError):
                # Compatibility fallback for light wrappers: obtain raw logits then activate once.
                values=self.model.predict(pairs);raw=[1/(1+math.exp(-float(v))) for v in values]
        elif self.activation=="identity":
            try:
                from torch.nn import Identity
                raw=self.model.predict(pairs,activation_fn=Identity())
            except (ImportError,TypeError):raw=self.model.predict(pairs)
        else:raw=self.model.predict(pairs)
        for hit,value in zip(candidates,raw):
            ce=self._score(value);hit.rerank_score=ce;hit.reasons.append(f"cross_encoder={ce:.4f}");hit.score=ce if self.blend is None else (1-self.blend)*hit.score+self.blend*ce
        return sorted(candidates,key=lambda h:h.score,reverse=True)[:top_k]

class CallableReranker:
    def __init__(self,score_fn,blend:float|None=None)->None:
        if blend is not None and not 0<=blend<=1:raise ValueError("blend must be between 0 and 1")
        self.score_fn=score_fn;self.blend=blend
    def rerank(self,query:str,hits:list[SearchHit],top_k:int)->list[SearchHit]:
        for hit in hits:
            score=float(self.score_fn(query,hit.document));hit.rerank_score=score;hit.score=score if self.blend is None else (1-self.blend)*hit.score+self.blend*score;hit.reasons.append(f"reranker={score:.4f}")
        return sorted(hits,key=lambda h:h.score,reverse=True)[:top_k]
