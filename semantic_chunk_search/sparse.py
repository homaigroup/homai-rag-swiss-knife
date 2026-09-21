from __future__ import annotations

from abc import ABC,abstractmethod
from collections import defaultdict
from collections.abc import Sequence

SparseVector=dict[int,float]

class BaseSparseEncoder(ABC):
    @abstractmethod
    def encode_query(self,text:str)->SparseVector:...
    @abstractmethod
    def encode_document(self,text:str)->SparseVector:...
    def encode_query_batch(self,texts:Sequence[str])->list[SparseVector]:return [self.encode_query(t) for t in texts]
    def encode_document_batch(self,texts:Sequence[str])->list[SparseVector]:return [self.encode_document(t) for t in texts]

class CallableSparseEncoder(BaseSparseEncoder):
    def __init__(self,query_fn,document_fn=None,query_batch_fn=None,document_batch_fn=None)->None:
        self.query_fn=query_fn;self.document_fn=document_fn or query_fn;self.query_batch_fn=query_batch_fn;self.document_batch_fn=document_batch_fn
    def encode_query(self,text:str)->SparseVector:return dict(self.query_fn(text))
    def encode_document(self,text:str)->SparseVector:return dict(self.document_fn(text))
    def encode_query_batch(self,texts):return [dict(x) for x in self.query_batch_fn(texts)] if self.query_batch_fn else super().encode_query_batch(texts)
    def encode_document_batch(self,texts):return [dict(x) for x in self.document_batch_fn(texts)] if self.document_batch_fn else super().encode_document_batch(texts)

class SentenceTransformerSparseEncoder(BaseSparseEncoder):
    def __init__(self,model_name:str="naver/splade-cocondenser-ensembledistil")->None:
        try:from sentence_transformers import SparseEncoder
        except (ImportError,AttributeError) as exc:raise ImportError("A sentence-transformers version with SparseEncoder support is required") from exc
        self.model=SparseEncoder(model_name)
    @staticmethod
    def _row_to_dict(row)->SparseVector:
        if hasattr(row,"tocoo"):
            coo=row.tocoo();return {int(i):float(v) for i,v in zip(coo.col,coo.data)}
        if hasattr(row,"indices") and hasattr(row,"values"):
            try:row=row.coalesce()
            except Exception:pass
            indices=row.indices();values=row.values()
            # Individual 1-D COO rows expose indices with shape [1, nnz].
            idx=indices[0] if getattr(indices,"ndim",1)>1 else indices
            return {int(i):float(v) for i,v in zip(idx.tolist(),values.tolist())}
        return {int(i):float(v) for i,v in enumerate(row) if float(v)!=0.0}
    def encode_query(self,text:str)->SparseVector:return self.encode_query_batch([text])[0]
    def encode_document(self,text:str)->SparseVector:return self.encode_document_batch([text])[0]
    def encode_query_batch(self,texts)->list[SparseVector]:return [self._row_to_dict(row) for row in self.model.encode_query(list(texts),convert_to_tensor=False,convert_to_sparse_tensor=True)]
    def encode_document_batch(self,texts)->list[SparseVector]:return [self._row_to_dict(row) for row in self.model.encode_document(list(texts),convert_to_tensor=False,convert_to_sparse_tensor=True)]

class InMemorySparseIndex:
    def __init__(self,encoder:BaseSparseEncoder)->None:self.encoder=encoder;self.vectors:dict[str,SparseVector]={};self.postings:dict[int,dict[str,float]]=defaultdict(dict)
    def _register(self,document_id:str,vec:SparseVector)->None:
        self.remove(document_id);self.vectors[document_id]=vec
        for dim,weight in vec.items():self.postings[int(dim)][document_id]=float(weight)
    def add(self,document_id:str,text:str)->None:self._register(document_id,self.encoder.encode_document(text))
    def add_many(self,items:list[tuple[str,str]])->None:
        if not items:return
        vecs=self.encoder.encode_document_batch([text for _,text in items])
        for (doc_id,_),vec in zip(items,vecs):self._register(doc_id,vec)
    def remove(self,document_id:str)->None:
        vec=self.vectors.pop(document_id,None)
        if not vec:return
        for dim in vec:
            bucket=self.postings.get(dim)
            if bucket is not None:
                bucket.pop(document_id,None)
                if not bucket:self.postings.pop(dim,None)
    def search(self,query:str,top_k:int,allowed_ids:set[str]|None=None)->list[tuple[str,float]]:
        qv=self.encoder.encode_query(query);scores:dict[str,float]=defaultdict(float)
        for dim,qweight in qv.items():
            for doc_id,dweight in self.postings.get(dim,{}).items():
                if allowed_ids is None or doc_id in allowed_ids:scores[doc_id]+=float(qweight)*dweight
        return sorted(scores.items(),key=lambda x:x[1],reverse=True)[:top_k]
