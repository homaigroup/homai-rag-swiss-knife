from __future__ import annotations

import re
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Protocol
from threading import RLock

from ..embedding import BaseEmbeddingProvider
from ..models import MultiVector, SearchDocument
from ..normalization import DEFAULT_NORMALIZER, TextNormalizer

_WORD_RE = re.compile(r"[\w\-]+", re.UNICODE)
_REFERENCE_RE = re.compile(r"(?:https?://\S+|\b[A-Z]{2,}[\-_]?\d{2,}\b|\b\w+[\-_]\d{3,}\b)", re.IGNORECASE)
def _component_signature(obj) -> str:
    if obj is None:return "none"
    value=getattr(obj,"signature",None)
    if value is not None:return str(value() if callable(value) else value)
    return f"{obj.__class__.__module__}.{obj.__class__.__qualname__}"

_STOPWORDS = {"the","a","an","and","or","of","to","in","on","for","with","is","are","this","that","from","by","be","as","at","it","its","و","در","به","از","که","را","با","برای","این","آن","یک","است","هست","می","شود","شد"}


class QueryLLM(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass(slots=True)
class QueryPerspectives:
    raw_text: str
    summary: str = ""
    keywords: str = ""
    titles: str = ""
    references: str = ""


class QueryEnricher:
    def __init__(self, embedder: BaseEmbeddingProvider, llm: QueryLLM | None = None, normalizer: TextNormalizer | None = None, cache_size: int = 1024) -> None:
        self.embedder, self.llm = embedder, llm
        self.normalizer = normalizer or DEFAULT_NORMALIZER
        self.cache_size=cache_size
        self._cache: OrderedDict[str, MultiVector] = OrderedDict(); self._lock=RLock()

    def _heuristic(self, query: str) -> QueryPerspectives:
        normalized = self.normalizer.normalize(query)
        words = _WORD_RE.findall(normalized.casefold()); unique=[]; seen=set()
        for word in words:
            if len(word)>2 and word not in _STOPWORDS and word not in seen: seen.add(word); unique.append(word)
        refs=_REFERENCE_RE.findall(normalized)
        # Titles are intentionally not empty anymore: noun-ish key phrases provide a candidate-generation view.
        titles=" ".join(unique[:6])
        return QueryPerspectives(normalized, normalized, ", ".join(unique[:16]), titles, " | ".join(refs))

    def perspectives(self, query: str) -> QueryPerspectives:
        base=self._heuristic(query)
        if self.llm is None: return base
        try:
            raw=self.llm.generate(
                "Return 4 lines exactly: SUMMARY:, KEYWORDS:, TITLES:, REFERENCES: for semantic search query: " + query
            )
            fields={}
            for line in raw.splitlines():
                if ":" in line:
                    k,v=line.split(":",1); fields[k.strip().lower()]=v.strip()
            return QueryPerspectives(base.raw_text, fields.get("summary") or base.summary, fields.get("keywords") or base.keywords, fields.get("titles") or base.titles, fields.get("references") or base.references)
        except Exception: return base

    def enrich(self, query: str) -> MultiVector:
        key=f"{self.embedder.signature}::{self.normalizer.normalize_for_search(query)}"
        with self._lock:
            if key in self._cache:
                value=self._cache.pop(key); self._cache[key]=value; return value
        p=self.perspectives(query)
        fields=[("raw_text",p.raw_text),("summary",p.summary),("keywords",p.keywords),("titles",p.titles),("references",p.references)]
        nonempty=[(n,t) for n,t in fields if t.strip()]; encoded=self.embedder.embed_query_batch([t for _,t in nonempty]); by={n:v for (n,_),v in zip(nonempty,encoded)}
        value=MultiVector(**{n:by.get(n,[]) for n,_ in fields})
        with self._lock:
            self._cache[key]=value
            while len(self._cache)>self.cache_size:self._cache.popitem(last=False)
        return value

    def clear_cache(self)->None:
        with self._lock:self._cache.clear()


class DocumentEnricher:
    def __init__(self, normalizer: TextNormalizer | None = None, contextualizer: QueryLLM | None = None) -> None:
        self.normalizer=normalizer or DEFAULT_NORMALIZER; self.contextualizer=contextualizer
    @property
    def signature(self)->str:
        import hashlib
        raw=f"document-enricher-v4|{self.normalizer.signature}|{_component_signature(self.contextualizer)}"
        return hashlib.blake2b(raw.encode(),digest_size=12).hexdigest()

    def enrich(self, document: SearchDocument) -> SearchDocument:
        text=self.normalizer.normalize(document.text.strip()); document.text=text
        if not document.summary:
            parts=re.split(r"(?<=[.!?؟])\s+|\n+",text); document.summary=" ".join(p.strip() for p in parts[:2] if p.strip())[:700] or text[:700]
        if not document.keywords:
            counts=Counter(w.casefold() for w in _WORD_RE.findall(text) if len(w)>2 and w.casefold() not in _STOPWORDS); document.keywords=[w for w,_ in counts.most_common(20)]
        if not document.titles:
            titles=[]
            mt=document.metadata.get("title") or document.metadata.get("document_title")
            if mt:
                titles.append(str(mt))
            titles.extend(document.section_path)
            document.titles=list(dict.fromkeys(t for t in titles if t))
        if not document.references: document.references=list(dict.fromkeys(_REFERENCE_RE.findall(text)))[:20]
        if not document.metadata.get("entities"):
            document.metadata["entities"]=list(dict.fromkeys([*document.titles,*document.references]))[:24]
        if not document.context_text:
            context=[]
            if document.titles: context.append("Title/section: "+" > ".join(document.titles))
            parent_context=document.metadata.get("parent_context")
            if parent_context: context.append("Parent context: "+str(parent_context))
            generated=""
            if self.contextualizer is not None:
                try:
                    generated=self.contextualizer.generate("In one concise sentence, state the document context needed to understand this chunk. Section: "+str(parent_context)+" Chunk: "+text[:1800]).strip()
                except Exception: generated=""
            if generated: context.append("Context: "+generated)
            context.append(text); document.context_text="\n".join(context)
        return document


class DocumentVectorizer:
    def __init__(self, embedder: BaseEmbeddingProvider, enricher: DocumentEnricher | None = None) -> None:
        self.embedder=embedder; self.enricher=enricher or DocumentEnricher()
    @property
    def signature(self)->str:
        import hashlib
        raw=f"document-vectorizer-v4|{self.embedder.signature}|{self.enricher.signature}"
        return hashlib.blake2b(raw.encode(),digest_size=12).hexdigest()
    def vectorize(self, document: SearchDocument) -> MultiVector:
        self.enricher.enrich(document)
        fields=[("raw_text",document.context_text or document.text),("summary",document.summary),("keywords",", ".join(document.keywords)),("titles"," | ".join(document.titles)),("references"," | ".join(document.references))]
        nonempty=[(n,t) for n,t in fields if t.strip()]; vectors=self.embedder.embed_document_batch([t for _,t in nonempty]); by={n:v for (n,_),v in zip(nonempty,vectors)}
        document.embedding_model=self.embedder.model_name
        document.embedding_dimensions=self.embedder.dimensions
        return MultiVector(**{n:by.get(n,[]) for n,_ in fields})

    def vectorize_many(self, documents: list[SearchDocument]) -> list[MultiVector]:
        """Batch all document/representation texts into one embedding call."""
        field_rows: list[list[tuple[str, str]]] = []
        flat: list[tuple[int, str, str]] = []
        for i, document in enumerate(documents):
            self.enricher.enrich(document)
            fields=[("raw_text",document.context_text or document.text),("summary",document.summary),("keywords",", ".join(document.keywords)),("titles"," | ".join(document.titles)),("references"," | ".join(document.references))]
            field_rows.append(fields)
            for name,text in fields:
                if text.strip(): flat.append((i,name,text))
        vectors=self.embedder.embed_document_batch([text for _,_,text in flat]) if flat else []
        mapped: list[dict[str,list[float]]] = [dict() for _ in documents]
        for (i,name,_), vec in zip(flat,vectors): mapped[i][name]=vec
        out=[]
        for document, row in zip(documents, mapped):
            document.embedding_model=self.embedder.model_name
            document.embedding_dimensions=self.embedder.dimensions
            out.append(MultiVector(**{name:row.get(name,[]) for name in ("raw_text","summary","keywords","titles","references")}))
        return out
