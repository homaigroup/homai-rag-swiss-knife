from __future__ import annotations

from abc import ABC,abstractmethod
from collections import defaultdict
from collections.abc import Callable
from threading import RLock

from ..advanced import SimpleEntityGraph
from ..embedding import cosine_similarity,l2_normalize
from ..models import SearchDocument
from ..sparse import BaseSparseEncoder,InMemorySparseIndex
from ..security import acl_allows
from .enrichment import DocumentVectorizer
from .lexical import BM25FIndex,ExactTermIndex

class BaseSemanticIndex(ABC):
    @abstractmethod
    def add(self,document:SearchDocument)->None:...
    def add_many(self,documents:list[SearchDocument])->None:
        for document in documents:self.add(document)
    @abstractmethod
    def remove(self,document_id:str)->bool:...
    @abstractmethod
    def clear(self)->None:...
    @abstractmethod
    def get(self,document_id:str)->SearchDocument|None:...
    def get_many(self,document_ids:list[str])->dict[str,SearchDocument]:
        return {doc_id:doc for doc_id in document_ids if (doc:=self.get(doc_id)) is not None}
    @abstractmethod
    def documents(self,filters:dict|None=None,predicate:Callable[[SearchDocument],bool]|None=None)->list[SearchDocument]:...
    def documents_by_source(self,source_id:str)->list[SearchDocument]:return [d for d in self.documents() if str(d.source_id)==str(source_id)]
    def ids_by_source(self,source_id:str)->set[str]:return {d.id for d in self.documents_by_source(source_id)}
    def remove_source(self,source_id:str)->int:
        ids=list(self.ids_by_source(source_id))
        for doc_id in ids:self.remove(doc_id)
        return len(ids)
    def replace_source(self,source_id:str,documents:list[SearchDocument])->None:
        self.remove_source(source_id);self.add_many(documents)
    def restore_source(self,source_id:str,documents:list[SearchDocument])->None:
        """Best-effort rollback hook. Backends may override with an independent recovery path."""
        self.replace_source(source_id,documents)
    def search_dense(self,query_vector:list[float],top_k:int,filters=None,predicate=None):return self.search_dense_dimension("raw_text",query_vector,top_k,filters,predicate)
    @abstractmethod
    def search_dense_dimension(self,dimension:str,query_vector:list[float],top_k:int,filters=None,predicate=None)->list[tuple[str,float]]:...
    @abstractmethod
    def search_lexical(self,query:str,top_k:int,filters=None,predicate=None)->list[tuple[str,float]]:...
    def search_sparse(self,query:str,top_k:int,filters=None,predicate=None)->list[tuple[str,float]]:return []
    def search_exact(self,query:str,top_k:int,filters=None,predicate=None)->list[tuple[str,float]]:return []
    def search_graph(self,query:str,top_k:int,filters=None,predicate=None,mode:str="drift")->list[tuple[str,float]]:return []

class InMemorySemanticIndex(BaseSemanticIndex):
    """Thread-safe exact multi-representation index; intended for tests/small corpora."""
    def __init__(self,vectorizer:DocumentVectorizer,sparse_encoder:BaseSparseEncoder|None=None,graph:SimpleEntityGraph|None=None)->None:
        self.vectorizer=vectorizer;self._documents:dict[str,SearchDocument]={};self._source_ids:dict[str,set[str]]=defaultdict(set);self._lock=RLock()
        self._bm25=BM25FIndex();self._exact=ExactTermIndex();self._sparse=InMemorySparseIndex(sparse_encoder) if sparse_encoder is not None else None;self._graph=graph
    def _register(self,document:SearchDocument)->None:
        self._documents[document.id]=document;self._source_ids[str(document.source_id or "")].add(document.id)
        fields={"body":document.context_text or document.text,"summary":document.summary,"keywords":" ".join(document.keywords),"titles":" ".join(document.titles),"references":" ".join(document.references)}
        self._bm25.add(document.id,fields);searchable="\n".join(filter(None,fields.values()));self._exact.add(document.id,searchable,document.references)
        if self._sparse is not None:self._sparse.add(document.id,searchable)
        if self._graph is not None:self._graph.add_document(document)
    def add(self,document:SearchDocument)->None:
        with self._lock:
            if document.id in self._documents:self.remove(document.id)
            if document.vectors is None:document.vectors=self.vectorizer.vectorize(document)
            self._register(document)
    def add_many(self,documents:list[SearchDocument])->None:
        with self._lock:
            missing=[d for d in documents if d.vectors is None]
            if missing:
                vectors=self.vectorizer.vectorize_many(missing)
                for d,v in zip(missing,vectors):d.vectors=v
            # Batch sparse document encoding when supported.
            sparse_items=[]
            for d in documents:
                if d.id in self._documents:self.remove(d.id)
                self._documents[d.id]=d;self._source_ids[str(d.source_id or "")].add(d.id)
                fields={"body":d.context_text or d.text,"summary":d.summary,"keywords":" ".join(d.keywords),"titles":" ".join(d.titles),"references":" ".join(d.references)}
                self._bm25.add(d.id,fields);searchable="\n".join(filter(None,fields.values()));self._exact.add(d.id,searchable,d.references)
                if self._sparse is not None:sparse_items.append((d.id,searchable))
                if self._graph is not None:self._graph.add_document(d)
            if self._sparse is not None and sparse_items:self._sparse.add_many(sparse_items)
    def remove(self,document_id:str)->bool:
        with self._lock:
            old=self._documents.pop(document_id,None)
            if old is None:return False
            bucket=self._source_ids.get(str(old.source_id or ""));
            if bucket is not None:
                bucket.discard(document_id)
                if not bucket:self._source_ids.pop(str(old.source_id or ""),None)
            self._bm25.remove(document_id);self._exact.remove(document_id)
            if self._sparse is not None:self._sparse.remove(document_id)
            if self._graph is not None:self._graph.remove_document(document_id)
            return True
    def clear(self)->None:
        with self._lock:
            self._documents.clear();self._source_ids.clear();self._bm25.clear();self._exact.clear()
            if self._sparse is not None:
                for doc_id in list(self._sparse.vectors):self._sparse.remove(doc_id)
            if self._graph is not None:self._graph=SimpleEntityGraph()
    def get(self,document_id:str)->SearchDocument|None:
        with self._lock:return self._documents.get(document_id)
    def get_many(self,document_ids:list[str])->dict[str,SearchDocument]:
        with self._lock:return {doc_id:self._documents[doc_id] for doc_id in document_ids if doc_id in self._documents}
    def documents(self,filters=None,predicate=None)->list[SearchDocument]:
        with self._lock:docs=list(self._documents.values())
        raw=dict(filters or {});security=raw.pop("__security__",None)
        if raw:docs=[d for d in docs if all(d.metadata.get(k)==v for k,v in raw.items())]
        if security:
            docs=[d for d in docs if acl_allows(d,principal=security.get("principal"),groups=security.get("groups"),tenant_id=security.get("tenant_id"),tenant_mode=security.get("tenant_mode","isolated"))]
        if predicate:docs=[d for d in docs if predicate(d)]
        return docs
    def documents_by_source(self,source_id:str)->list[SearchDocument]:
        with self._lock:return [self._documents[i] for i in list(self._source_ids.get(str(source_id),set())) if i in self._documents]
    def ids_by_source(self,source_id:str)->set[str]:
        with self._lock:return set(self._source_ids.get(str(source_id),set()))
    def replace_source(self,source_id:str,documents:list[SearchDocument])->None:
        # One in-process critical section: readers never observe an empty half-replaced source.
        with self._lock:
            self.remove_source(source_id);self.add_many(documents)
    def restore_source(self,source_id:str,documents:list[SearchDocument])->None:
        # Recovery intentionally does not depend on remove_source/replace_source: if an
        # update failed halfway through one of those methods, rebuild the local structures
        # from the surviving non-source documents plus the known-good snapshot.
        with self._lock:
            others=[d for d in self._documents.values() if str(d.source_id or "")!=str(source_id)]
            self.clear();self.add_many([*others,*documents])
    def _allowed(self,filters,predicate)->set[str]:return {d.id for d in self.documents(filters,predicate)}
    def search_dense_dimension(self,dimension,query_vector,top_k,filters=None,predicate=None):
        scored=[]
        for doc in self.documents(filters,predicate):
            dv=getattr(doc.vectors,dimension,[]) if doc.vectors else []
            if not dv or not query_vector:continue
            score=max(0.0,cosine_similarity(query_vector,dv))
            if score>0:scored.append((doc.id,score))
        return sorted(scored,key=lambda x:x[1],reverse=True)[:top_k]
    def search_lexical(self,query,top_k,filters=None,predicate=None):
        with self._lock:return self._bm25.search(query,top_k,self._allowed(filters,predicate))
    def search_sparse(self,query,top_k,filters=None,predicate=None):
        with self._lock:return self._sparse.search(query,top_k,self._allowed(filters,predicate)) if self._sparse is not None else []
    def search_exact(self,query,top_k,filters=None,predicate=None):
        with self._lock:return self._exact.search(query,top_k,self._allowed(filters,predicate))
    def search_graph(self,query,top_k,filters=None,predicate=None,mode="drift"):
        with self._lock:
            if self._graph is None:return []
            allowed=self._allowed(filters,predicate);ids=[x for x in self._graph.search(query,top_k*2,mode=mode) if x in allowed][:top_k]
            return [(doc_id,1.0/(i+1)) for i,doc_id in enumerate(ids)]
    def context_window(self,document_id:str,neighbors:int=1)->list[SearchDocument]:
        doc=self.get(document_id)
        if doc is None:return []
        same=[d for d in self.documents_by_source(str(doc.source_id)) if d.chunk_index is not None];same.sort(key=lambda d:int(d.chunk_index or 0))
        try:pos=next(i for i,d in enumerate(same) if d.id==document_id)
        except StopIteration:return [doc]
        return same[max(0,pos-neighbors):min(len(same),pos+neighbors+1)]
    def __len__(self)->int:
        with self._lock:return len(self._documents)

class HNSWSemanticIndex(InMemorySemanticIndex):
    """Filter-aware HNSW over every named dense representation, with exact stores alongside it."""
    DIMS=("raw_text","summary","keywords","titles","references")
    def __init__(self,vectorizer:DocumentVectorizer,*,dimensions:int,ann_dimensions:int|None=None,max_elements:int=100_000,ef_construction:int=200,m:int=16,ef_search:int=100,sparse_encoder=None,graph=None)->None:
        try:import hnswlib
        except ImportError as exc:raise ImportError("Install hnswlib or semantic-chunk-search[ann]") from exc
        super().__init__(vectorizer,sparse_encoder=sparse_encoder,graph=graph);self._hnswlib=hnswlib;self._dimensions=dimensions;self._ann_dimensions=int(ann_dimensions or dimensions);
        if self._ann_dimensions<1 or self._ann_dimensions>dimensions:raise ValueError("ann_dimensions must be within embedding dimensions")
        self._max_elements=max_elements;self._ef_construction=ef_construction;self._m=m;self._ef_search=ef_search
        self._anns={d:self._new_ann() for d in self.DIMS};self._id_to_label={};self._label_to_id={};self._next_label=0
    def _new_ann(self):
        idx=self._hnswlib.Index(space="cosine",dim=self._ann_dimensions);idx.init_index(max_elements=self._max_elements,ef_construction=self._ef_construction,M=self._m);idx.set_ef(self._ef_search);return idx
    def _ensure_capacity(self):
        if self._next_label>=self._max_elements:
            self._max_elements=max(self._max_elements*2,self._next_label+1)
            for ann in self._anns.values():ann.resize_index(self._max_elements)
    def add(self,document:SearchDocument)->None:
        with self._lock:
            if document.id in self._id_to_label:
                old=self._id_to_label.pop(document.id);self._label_to_id.pop(old,None)
                for ann in self._anns.values():
                    try:ann.mark_deleted(old)
                    except RuntimeError:pass
            super().add(document);self._ensure_capacity();label=self._next_label;self._next_label+=1;self._id_to_label[document.id]=label;self._label_to_id[label]=document.id
            if document.vectors:
                for dim in self.DIMS:
                    vector=getattr(document.vectors,dim,[])
                    if vector:
                        projected=l2_normalize(vector[:self._ann_dimensions]) if self._ann_dimensions<len(vector) else vector
                        self._anns[dim].add_items([projected],[label])
    def add_many(self,documents:list[SearchDocument])->None:
        with self._lock:
            missing=[d for d in documents if d.vectors is None]
            if missing:
                for d,v in zip(missing,self.vectorizer.vectorize_many(missing)):d.vectors=v
            for d in documents:self.add(d)
    def remove(self,document_id:str)->bool:
        with self._lock:
            label=self._id_to_label.pop(document_id,None)
            if label is not None:
                self._label_to_id.pop(label,None)
                for ann in self._anns.values():
                    try:ann.mark_deleted(label)
                    except RuntimeError:pass
            return super().remove(document_id)
    def clear(self)->None:
        with self._lock:
            super().clear();self._anns={d:self._new_ann() for d in self.DIMS};self._id_to_label.clear();self._label_to_id.clear();self._next_label=0
    def search_dense_dimension(self,dimension,query_vector,top_k,filters=None,predicate=None):
        if dimension not in self._anns or not query_vector:return super().search_dense_dimension(dimension,query_vector,top_k,filters,predicate)
        with self._lock:
            allowed_ids=self._allowed(filters,predicate) if (filters or predicate) else set(self._id_to_label)
            allowed_labels={self._id_to_label[x] for x in allowed_ids if x in self._id_to_label}
            if not allowed_labels:return []
            prefetch=min(len(allowed_labels),max(top_k,top_k*10 if self._ann_dimensions<self._dimensions else top_k))
            qproj=l2_normalize(query_vector[:self._ann_dimensions]) if self._ann_dimensions<len(query_vector) else query_vector
            # hnswlib's label filter keeps ANN traversal active instead of falling back to O(N) exact scan.
            labels,distances=self._anns[dimension].knn_query([qproj],k=prefetch,filter=lambda label:int(label) in allowed_labels)
            out=[]
            for label,distance in zip(labels[0],distances[0]):
                doc_id=self._label_to_id.get(int(label))
                if doc_id is None:continue
                if self._ann_dimensions<self._dimensions:
                    doc=self._documents.get(doc_id);dv=getattr(doc.vectors,dimension,[]) if doc and doc.vectors else []
                    score=max(0.0,cosine_similarity(query_vector,dv)) if dv else 0.0
                else:score=max(0.0,1.0-float(distance))
                out.append((doc_id,score))
            return sorted(out,key=lambda x:x[1],reverse=True)[:top_k]


class MatryoshkaHNSWSemanticIndex(HNSWSemanticIndex):
    """Two-stage MRL index: short HNSW vectors for candidate generation, full vectors for exact rescoring.

    The underlying embedding model must be trained for Matryoshka Representation Learning.
    """
    def __init__(self,vectorizer:DocumentVectorizer,*,dimensions:int,coarse_dimensions:int=128,**kwargs)->None:
        super().__init__(vectorizer,dimensions=dimensions,ann_dimensions=coarse_dimensions,**kwargs)
