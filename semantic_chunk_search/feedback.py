from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from threading import RLock

from .normalization import DEFAULT_NORMALIZER

@dataclass(slots=True)
class FeedbackEvent:
    query:str;document_id:str;relevance:float;kind:str="explicit"

class FeedbackStore:
    def __init__(self)->None:
        self.events:list[FeedbackEvent]=[];self._scores:dict[tuple[str,str],list[float]]=defaultdict(list);self._lock=RLock()
    def record(self,query:str,document_id:str,relevance:float,kind:str="explicit")->None:
        q=DEFAULT_NORMALIZER.normalize_for_search(query);rel=max(-1.0,min(1.0,float(relevance)))
        with self._lock:
            self.events.append(FeedbackEvent(q,document_id,rel,kind));self._scores[(q,document_id)].append(rel)
    def score(self,query:str,document_id:str)->float:
        with self._lock:
            vals=list(self._scores.get((DEFAULT_NORMALIZER.normalize_for_search(query),document_id),[]))
        return sum(vals)/len(vals) if vals else 0.0
    def remove_documents(self,document_ids:set[str])->None:
        if not document_ids:return
        with self._lock:
            self.events=[e for e in self.events if e.document_id not in document_ids]
            for key in list(self._scores):
                if key[1] in document_ids:self._scores.pop(key,None)

class LinearLTRReranker:
    """Regularized pointwise feedback scorer; kept for backward compatibility.

    This is intentionally named LTR in the public API, but behaves as a calibrated
    pointwise logistic ranker. For LambdaMART/listwise ranking use a custom Reranker.
    """
    FEATURES=("semantic_score","dense_score","lexical_score","sparse_score","exact_score","graph_score","fusion_score")
    def __init__(self,learning_rate:float=0.05,epochs:int=150,l2:float=1e-3)->None:
        self.learning_rate=learning_rate;self.epochs=epochs;self.l2=l2;self.weights={f:0.0 for f in self.FEATURES};self.bias=0.0
    @staticmethod
    def _sigmoid(x:float)->float:
        import math
        if x>=0:z=math.exp(-x);return 1/(1+z)
        z=math.exp(x);return z/(1+z)
    @staticmethod
    def _label(v:float)->float:
        # Accept both [-1,1] feedback and [0,1] labels.
        v=float(v)
        return (v+1.0)/2.0 if v<0 else max(0.0,min(1.0,v))
    def fit(self,rows:list[tuple[dict[str,float],float]])->None:
        if not rows:return
        for _ in range(self.epochs):
            for features,label in rows:
                y=self._label(label);x=self.bias+sum(self.weights[f]*float(features.get(f,0.0)) for f in self.FEATURES);pred=self._sigmoid(x);err=y-pred
                self.bias+=self.learning_rate*err
                for f in self.FEATURES:
                    val=float(features.get(f,0.0));self.weights[f]+=self.learning_rate*(err*val-self.l2*self.weights[f])
    def rerank(self,query:str,hits:list,top_k:int):
        for hit in hits:
            features={f:float(getattr(hit,f,0.0) or 0.0) for f in self.FEATURES};score=self._sigmoid(self.bias+sum(self.weights[f]*features[f] for f in self.FEATURES))
            hit.rerank_score=score;hit.score=score;hit.reasons.append(f"ltr={score:.4f}")
        return sorted(hits,key=lambda h:h.score,reverse=True)[:top_k]
    def save(self,path:str|Path)->None:
        Path(path).write_text(json.dumps({"weights":self.weights,"bias":self.bias,"features":self.FEATURES},sort_keys=True),encoding="utf-8")
    def load(self,path:str|Path)->None:
        data=json.loads(Path(path).read_text(encoding="utf-8"));self.weights={f:float(data["weights"].get(f,0.0)) for f in self.FEATURES};self.bias=float(data.get("bias",0.0))


class PairwiseLinearLTRReranker:
    """Dependency-free pairwise linear ranker trained on preference pairs.

    Each training row is ``(positive_features, negative_features, weight)``. The
    model optimizes a logistic preference objective over feature differences, which
    is ranking-aware unlike the backward-compatible pointwise scorer above.
    """
    FEATURES=LinearLTRReranker.FEATURES
    def __init__(self,learning_rate:float=0.03,epochs:int=200,l2:float=1e-3)->None:
        self.learning_rate=learning_rate;self.epochs=epochs;self.l2=l2;self.weights={f:0.0 for f in self.FEATURES}
    @staticmethod
    def _sigmoid(x:float)->float:return LinearLTRReranker._sigmoid(x)
    def fit_pairs(self,rows:list[tuple[dict[str,float],dict[str,float],float|int]])->None:
        if not rows:return
        for _ in range(self.epochs):
            for pos,neg,weight in rows:
                diff={f:float(pos.get(f,0.0))-float(neg.get(f,0.0)) for f in self.FEATURES};margin=sum(self.weights[f]*diff[f] for f in self.FEATURES);err=(1.0-self._sigmoid(margin))*max(0.0,float(weight))
                for f in self.FEATURES:self.weights[f]+=self.learning_rate*(err*diff[f]-self.l2*self.weights[f])
    def score_features(self,features:dict[str,float])->float:
        return sum(self.weights[f]*float(features.get(f,0.0)) for f in self.FEATURES)
    def rerank(self,query:str,hits:list,top_k:int):
        for hit in hits:
            features={f:float(getattr(hit,f,0.0) or 0.0) for f in self.FEATURES};raw=self.score_features(features);score=self._sigmoid(raw);hit.rerank_score=score;hit.score=score;hit.reasons.append(f"pairwise_ltr={score:.4f}")
        return sorted(hits,key=lambda h:h.score,reverse=True)[:top_k]
    def save(self,path:str|Path)->None:Path(path).write_text(json.dumps({"weights":self.weights,"features":self.FEATURES},sort_keys=True),encoding="utf-8")
    def load(self,path:str|Path)->None:
        data=json.loads(Path(path).read_text(encoding="utf-8"));self.weights={f:float(data["weights"].get(f,0.0)) for f in self.FEATURES}
