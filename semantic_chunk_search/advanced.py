from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from threading import RLock
from typing import Callable, Iterable, Sequence

from .embedding import BaseEmbeddingProvider, cosine_similarity, l2_normalize
from .models import SearchDocument, SearchHit


class MatryoshkaEmbeddingProvider(BaseEmbeddingProvider):
    """Truncate an existing normalized embedding provider to a smaller dimension."""

    def __init__(self, base: BaseEmbeddingProvider, dimensions: int) -> None:
        if dimensions < 1 or dimensions > base.dimensions:
            raise ValueError("dimensions must be within base embedding dimensions")
        self.base = base
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return f"{self.base.model_name}:matryoshka-{self._dimensions}"

    def _cut(self, vector: Sequence[float]) -> list[float]:
        return l2_normalize(vector[: self._dimensions])

    def embed(self, text: str) -> list[float]:
        return self._cut(self.base.embed(text))

    def embed_query(self, text: str) -> list[float]:
        return self._cut(self.base.embed_query(text))

    def embed_document(self, text: str) -> list[float]:
        return self._cut(self.base.embed_document(text))

    def embed_query_batch(self, texts):
        return [self._cut(v) for v in self.base.embed_query_batch(texts)]

    def embed_document_batch(self, texts):
        return [self._cut(v) for v in self.base.embed_document_batch(texts)]


def quantize_int8(vector: Sequence[float]) -> tuple[list[int], float]:
    if not vector:
        return [], 1.0
    max_abs = max(abs(float(v)) for v in vector) or 1.0
    scale = max_abs / 127.0
    return [max(-127, min(127, int(round(float(v) / scale)))) for v in vector], scale


def dequantize_int8(values: Sequence[int], scale: float) -> list[float]:
    return [float(v) * scale for v in values]


class TokenMaxSimReranker:
    """Dependency-free late interaction approximation using the configured embedder per token/term."""

    def __init__(self, embedder: BaseEmbeddingProvider, max_query_terms: int = 24, max_document_terms: int = 96) -> None:
        self.embedder = embedder
        self.max_query_terms = max_query_terms
        self.max_document_terms = max_document_terms

    @staticmethod
    def _terms(text: str, limit: int) -> list[str]:
        return [t for t in text.split() if t][:limit]

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        qterms = self._terms(query, self.max_query_terms)
        qvecs = self.embedder.embed_query_batch(qterms) if qterms else []
        for hit in hits:
            dterms = self._terms(hit.document.text, self.max_document_terms)
            dvecs = self.embedder.embed_document_batch(dterms) if dterms else []
            if not qvecs or not dvecs:
                score = 0.0
            else:
                score = sum(max(max(0.0, cosine_similarity(qv, dv)) for dv in dvecs) for qv in qvecs) / len(qvecs)
            hit.late_interaction_score = score
            hit.score = 0.65 * hit.score + 0.35 * score
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]


@dataclass(slots=True)
class GraphEdge:
    source: str
    relation: str
    target: str
    document_id: str


class SimpleEntityGraph:
    """Small optional graph retriever for entity-centric/multi-hop workloads."""

    def __init__(self) -> None:
        self.edges: list[GraphEdge] = []
        self.entity_docs: dict[str, set[str]] = defaultdict(set)

    def add_document(self, document: SearchDocument) -> None:
        entities = [str(x).casefold() for x in document.metadata.get("entities", [])]
        for entity in entities:
            self.entity_docs[entity].add(document.id)
        for raw in document.metadata.get("relations", []):
            if isinstance(raw, (list, tuple)) and len(raw) >= 3:
                edge = GraphEdge(str(raw[0]).casefold(), str(raw[1]), str(raw[2]).casefold(), document.id)
                self.edges.append(edge)
                self.entity_docs[edge.source].add(document.id)
                self.entity_docs[edge.target].add(document.id)

    def remove_document(self, document_id: str) -> None:
        self.edges = [e for e in self.edges if e.document_id != document_id]
        for docs in self.entity_docs.values():
            docs.discard(document_id)

    def _neighbors(self, entity: str) -> set[str]:
        out=set()
        for edge in self.edges:
            if edge.source==entity: out.add(edge.target)
            elif edge.target==entity: out.add(edge.source)
        return out

    def communities(self) -> list[set[str]]:
        unseen=set(self.entity_docs); groups=[]
        while unseen:
            root=unseen.pop(); group={root}; stack=[root]
            while stack:
                current=stack.pop()
                for neighbor in self._neighbors(current):
                    if neighbor not in group:
                        group.add(neighbor); unseen.discard(neighbor); stack.append(neighbor)
            groups.append(group)
        return groups

    def search(self, query: str, top_k: int = 20, mode: str = "drift") -> list[str]:
        q=query.casefold(); scored:dict[str,float]=defaultdict(float); matched=[entity for entity in self.entity_docs if entity in q]
        if mode=="global":
            query_terms=set(q.split())
            ranked=[]
            for community in self.communities():
                overlap=sum(1 for entity in community if query_terms & set(entity.split()))
                importance=sum(len(self.entity_docs.get(entity,set())) for entity in community)
                ranked.append((overlap+0.01*importance,community))
            for score,community in sorted(ranked,key=lambda x:x[0],reverse=True):
                if score<=0:continue
                for entity in community:
                    for doc_id in self.entity_docs.get(entity,set()): scored[doc_id]+=score
        else:
            for entity in matched:
                for doc_id in self.entity_docs[entity]: scored[doc_id]+=2.0
                if mode in {"local","drift"}:
                    frontier=self._neighbors(entity)
                    for neighbor in frontier:
                        for doc_id in self.entity_docs.get(neighbor,set()): scored[doc_id]+=1.0
                    if mode=="drift":
                        for neighbor in list(frontier):
                            for second in self._neighbors(neighbor):
                                for doc_id in self.entity_docs.get(second,set()): scored[doc_id]+=0.4
        return [doc_id for doc_id,_ in sorted(scored.items(),key=lambda x:x[1],reverse=True)[:top_k]]


class MultimodalEmbeddingAdapter:
    """Bring external image/page embedding functions into the same retrieval architecture."""

    def __init__(self, embed_image: Callable[[object], list[float]], embed_text: Callable[[str], list[float]]) -> None:
        self.embed_image = embed_image
        self.embed_text = embed_text


@dataclass(slots=True)
class MultimodalPage:
    id: str
    image: object
    metadata: dict
    vector: list[float] | None = None


class MultimodalPageRetriever:
    """Optional visual-page retrieval using caller-provided image/text embedding functions."""
    def __init__(self, adapter: MultimodalEmbeddingAdapter) -> None:
        self.adapter=adapter; self.pages:dict[str,MultimodalPage]={}

    def add_page(self,page_id:str,image:object,metadata:dict|None=None)->None:
        self.pages[page_id]=MultimodalPage(page_id,image,dict(metadata or {}),self.adapter.embed_image(image))

    def search(self,query:str,top_k:int=10)->list[tuple[MultimodalPage,float]]:
        qv=self.adapter.embed_text(query); scored=[]
        for page in self.pages.values():
            if page.vector:
                scored.append((page,max(0.0,cosine_similarity(qv,page.vector))))
        scored.sort(key=lambda x:x[1],reverse=True); return scored[:top_k]


@dataclass(slots=True)
class MultiVectorPage:
    id: str
    vectors: list[list[float]]
    metadata: dict


class MultiVectorPageRetriever:
    """ColPali/ColQwen-style page retrieval via MaxSim over multiple page vectors."""
    def __init__(self, query_encoder: Callable[[str], list[list[float]]]) -> None:
        self.query_encoder=query_encoder
        self.pages:dict[str,MultiVectorPage]={}

    def add_page(self,page_id:str,vectors:list[list[float]],metadata:dict|None=None)->None:
        self.pages[page_id]=MultiVectorPage(page_id,vectors,dict(metadata or {}))

    @staticmethod
    def _maxsim(query_vectors:list[list[float]],document_vectors:list[list[float]])->float:
        if not query_vectors or not document_vectors:return 0.0
        return sum(max(max(0.0,cosine_similarity(q,d)) for d in document_vectors) for q in query_vectors)/len(query_vectors)

    def search(self,query:str,top_k:int=10)->list[tuple[MultiVectorPage,float]]:
        qv=self.query_encoder(query); scored=[(p,self._maxsim(qv,p.vectors)) for p in self.pages.values()]
        scored.sort(key=lambda x:x[1],reverse=True); return scored[:top_k]

class BaseLateInteractionEncoder:
    """ColBERT-compatible token-matrix contract.

    Implementations return contextual token vectors for the whole query/document,
    rather than embedding isolated whitespace terms.
    """
    def encode_query(self,text:str)->list[list[float]]:raise NotImplementedError
    def encode_document(self,text:str)->list[list[float]]:raise NotImplementedError

class CallableLateInteractionEncoder(BaseLateInteractionEncoder):
    def __init__(self,query_fn,document_fn=None)->None:self.query_fn=query_fn;self.document_fn=document_fn or query_fn
    def encode_query(self,text:str)->list[list[float]]:return [list(v) for v in self.query_fn(text)]
    def encode_document(self,text:str)->list[list[float]]:return [list(v) for v in self.document_fn(text)]

class MultiVectorLateInteractionReranker:
    """True late-interaction MaxSim over contextual query/document token matrices."""
    def __init__(self,encoder:BaseLateInteractionEncoder,blend:float=0.35,cache_size:int=4096)->None:
        from collections import OrderedDict
        if not 0<=blend<=1:raise ValueError("blend must be between 0 and 1")
        self.encoder=encoder;self.blend=blend;self.cache_size=cache_size;self._cache=OrderedDict();self._lock=RLock()
    @staticmethod
    def _maxsim(qv,dv)->float:
        if not qv or not dv:return 0.0
        return sum(max(max(0.0,cosine_similarity(q,d)) for d in dv) for q in qv)/len(qv)
    def _doc_vectors(self,hit):
        key=(hit.document.id,hit.document.content_hash)
        with self._lock:
            if key in self._cache:
                value=self._cache.pop(key);self._cache[key]=value;return value
        value=self.encoder.encode_document(hit.document.context_text or hit.document.text)
        with self._lock:
            self._cache[key]=value
            while len(self._cache)>self.cache_size:self._cache.popitem(last=False)
        return value
    def rerank(self,query:str,hits:list[SearchHit],top_k:int)->list[SearchHit]:
        qv=self.encoder.encode_query(query)
        for hit in hits:
            score=self._maxsim(qv,self._doc_vectors(hit));hit.late_interaction_score=score;hit.score=(1-self.blend)*hit.score+self.blend*score;hit.reasons.append(f"late_interaction={score:.4f}")
        return sorted(hits,key=lambda h:h.score,reverse=True)[:top_k]


class GraphRetrieverAdapter:
    """Adapter for an external/full GraphRAG implementation.

    ``search_fn`` receives ``(query, top_k, mode)`` and returns document IDs or
    ``(document_id, score)`` pairs. This lets the core use real Local/Global/DRIFT
    graph systems without pretending the built-in metadata graph is full GraphRAG.
    """
    def __init__(self,search_fn,add_fn=None,remove_fn=None)->None:self.search_fn=search_fn;self.add_fn=add_fn;self.remove_fn=remove_fn
    def add_document(self,document:SearchDocument)->None:
        if self.add_fn:self.add_fn(document)
    def remove_document(self,document_id:str)->None:
        if self.remove_fn:self.remove_fn(document_id)
    def search(self,query:str,top_k:int=20,mode:str="drift")->list[str]:
        rows=list(self.search_fn(query,top_k,mode) or []);return [str(x[0] if isinstance(x,(tuple,list)) else x) for x in rows[:top_k]]
