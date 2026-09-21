from __future__ import annotations

from dataclasses import asdict
import json
import re
from typing import Iterable
from uuid import NAMESPACE_URL,uuid5

from ..models import MultiVector,SearchDocument
from ..normalization import DEFAULT_NORMALIZER
from ..sparse import BaseSparseEncoder
from .enrichment import DocumentVectorizer
from .index import BaseSemanticIndex

_REFERENCE_RE=re.compile(r"(?:https?://\S+|\b[A-Z]{2,}[\-_]?\d{2,}\b|\b\w+[\-_]\d{3,}\b)",re.I)

class QdrantSemanticIndex(BaseSemanticIndex):
    """Concrete Qdrant backend using named dense vectors and optional sparse vectors.

    It avoids keeping the corpus in Python RAM. Dense representations are stored as
    named vectors; source-scoped upserts avoid an empty replacement window. Arbitrary
    Python predicates are applied defensively after server retrieval, while simple
    metadata filters are pushed into Qdrant.
    """
    DIMS=("raw_text","summary","keywords","titles","references")
    def __init__(self,client,collection_name:str,vectorizer:DocumentVectorizer,*,dimensions:int,sparse_encoder:BaseSparseEncoder|None=None,create_collection:bool=False,create_payload_indexes:bool=False,wait:bool=True,overfetch:int=5)->None:
        try:from qdrant_client import models
        except ImportError as exc:raise ImportError("Install qdrant-client or semantic-chunk-search[qdrant]") from exc
        self.client=client;self.collection_name=collection_name;self.vectorizer=vectorizer;self.dimensions=dimensions;self.sparse_encoder=sparse_encoder;self.models=models;self.wait=wait;self.overfetch=max(1,int(overfetch))
        if create_collection and not self.client.collection_exists(collection_name):
            vectors={d:models.VectorParams(size=dimensions,distance=models.Distance.COSINE) for d in self.DIMS}
            sparse={"sparse":models.SparseVectorParams()} if sparse_encoder is not None else None
            kwargs={"collection_name":collection_name,"vectors_config":vectors}
            if sparse is not None:kwargs["sparse_vectors_config"]=sparse
            self.client.create_collection(**kwargs)
        if create_payload_indexes and hasattr(self.client,"create_payload_index"):
            schema=getattr(models,"PayloadSchemaType",None);keyword=getattr(schema,"KEYWORD","keyword") if schema is not None else "keyword"
            for field in ("source_id","tenant_id","visibility","owner","allowed_users","allowed_groups","references_normalized"):
                try:self.client.create_payload_index(collection_name=collection_name,field_name=field,field_schema=keyword,wait=wait)
                except Exception:pass

    @staticmethod
    def _point_id(document_id:str)->str:return str(uuid5(NAMESPACE_URL,"semantic-chunk-search:"+document_id))
    def _payload(self,doc:SearchDocument)->dict:
        data=asdict(doc);data.pop("vectors",None)
        meta=doc.metadata
        return {"document_id":doc.id,"source_id":str(doc.source_id or ""),"tenant_id":meta.get("tenant_id"),"visibility":meta.get("visibility","public"),"owner":meta.get("owner"),"allowed_users":list(meta.get("allowed_users",[]) or []),"allowed_groups":list(meta.get("allowed_groups",[]) or []),"references":list(doc.references or []),"references_normalized":[DEFAULT_NORMALIZER.normalize_for_search(x) for x in (doc.references or [])],"metadata":meta,"document":json.dumps(data,ensure_ascii=False,separators=(",",":"),default=str)}
    def _from_point(self,point)->SearchDocument:
        payload=dict(getattr(point,"payload",{}) or {});raw=payload.get("document")
        data=json.loads(raw) if isinstance(raw,str) else dict(raw or {})
        allowed=set(SearchDocument.__dataclass_fields__);data={k:v for k,v in data.items() if k in allowed}
        vectors=getattr(point,"vector",None)
        if isinstance(vectors,dict):data["vectors"]=MultiVector(**{d:list(vectors.get(d) or []) for d in self.DIMS})
        return SearchDocument(**data)
    def _filter(self,filters:dict|None=None,source_id:str|None=None,extra_must=None):
        raw=dict(filters or {});security=raw.pop("__security__",None);must=[];m=self.models
        if source_id is not None:must.append(m.FieldCondition(key="source_id",match=m.MatchValue(value=str(source_id))))
        for key,value in raw.items():must.append(m.FieldCondition(key=f"metadata.{key}",match=m.MatchValue(value=value)))
        if security:
            tenant_mode=security.get("tenant_mode","isolated");tenant_id=security.get("tenant_id")
            empty_cls=getattr(m,"IsEmptyCondition",None);payload_cls=getattr(m,"PayloadField",None)
            def empty(key):return empty_cls(is_empty=payload_cls(key=key)) if empty_cls and payload_cls else None
            if tenant_mode in {"isolated","shared_public"}:
                if tenant_id is None and tenant_mode=="isolated":
                    condition=empty("tenant_id")
                    if condition is not None:must.append(condition)
                elif tenant_id is not None:
                    choices=[m.FieldCondition(key="tenant_id",match=m.MatchValue(value=str(tenant_id)))]
                    condition=empty("tenant_id")
                    if condition is not None:choices.append(condition)
                    must.append(m.Filter(should=choices))
            access=[m.FieldCondition(key="visibility",match=m.MatchValue(value="public"))]
            principal=security.get("principal");groups=list(security.get("groups") or [])
            if principal is not None:
                access.append(m.FieldCondition(key="owner",match=m.MatchValue(value=str(principal))))
                access.append(m.FieldCondition(key="allowed_users",match=m.MatchValue(value=str(principal))))
            if groups:
                if hasattr(m,"MatchAny"):access.append(m.FieldCondition(key="allowed_groups",match=m.MatchAny(any=[str(g) for g in groups])))
                else:access.extend(m.FieldCondition(key="allowed_groups",match=m.MatchValue(value=str(g))) for g in groups)
            must.append(m.Filter(should=access))
        must.extend(list(extra_must or []))
        return m.Filter(must=must) if must else None
    def _point(self,doc:SearchDocument):
        if doc.vectors is None:doc.vectors=self.vectorizer.vectorize(doc)
        vector={d:getattr(doc.vectors,d) for d in self.DIMS if getattr(doc.vectors,d)}
        if self.sparse_encoder is not None:
            text="\n".join(filter(None,[doc.context_text or doc.text,doc.summary," ".join(doc.keywords)," ".join(doc.titles)," ".join(doc.references)]));sv=self.sparse_encoder.encode_document(text)
            vector["sparse"]=self.models.SparseVector(indices=list(sv),values=[sv[i] for i in sv])
        return self.models.PointStruct(id=self._point_id(doc.id),vector=vector,payload=self._payload(doc))
    def add(self,document:SearchDocument)->None:self.client.upsert(collection_name=self.collection_name,points=[self._point(document)],wait=self.wait)
    def add_many(self,documents:list[SearchDocument])->None:
        if not documents:return
        missing=[d for d in documents if d.vectors is None]
        if missing:
            for d,v in zip(missing,self.vectorizer.vectorize_many(missing)):d.vectors=v
        self.client.upsert(collection_name=self.collection_name,points=[self._point(d) for d in documents],wait=self.wait)
    def remove(self,document_id:str)->bool:
        self.client.delete(collection_name=self.collection_name,points_selector=[self._point_id(document_id)],wait=self.wait);return True
    def clear(self)->None:
        self.client.delete(collection_name=self.collection_name,points_selector=self.models.FilterSelector(filter=self.models.Filter()),wait=self.wait)
    def get(self,document_id:str)->SearchDocument|None:
        points=self.client.retrieve(collection_name=self.collection_name,ids=[self._point_id(document_id)],with_payload=True,with_vectors=True)
        return self._from_point(points[0]) if points else None
    def get_many(self,document_ids:list[str])->dict[str,SearchDocument]:
        if not document_ids:return {}
        points=self.client.retrieve(collection_name=self.collection_name,ids=[self._point_id(i) for i in document_ids],with_payload=True,with_vectors=True)
        docs=[self._from_point(p) for p in points];return {d.id:d for d in docs}
    def _scroll(self,query_filter=None)->list[SearchDocument]:
        out=[];offset=None
        while True:
            points,offset=self.client.scroll(collection_name=self.collection_name,scroll_filter=query_filter,limit=256,offset=offset,with_payload=True,with_vectors=True)
            out.extend(self._from_point(p) for p in points)
            if offset is None:break
        return out
    def documents(self,filters=None,predicate=None)->list[SearchDocument]:
        docs=self._scroll(self._filter(filters))
        return [d for d in docs if predicate(d)] if predicate else docs
    def documents_by_source(self,source_id:str)->list[SearchDocument]:return self._scroll(self._filter(source_id=source_id))
    def ids_by_source(self,source_id:str)->set[str]:return {d.id for d in self.documents_by_source(source_id)}
    def remove_source(self,source_id:str)->int:
        ids=self.ids_by_source(source_id)
        if ids:self.client.delete(collection_name=self.collection_name,points_selector=self.models.FilterSelector(filter=self._filter(source_id=source_id)),wait=self.wait)
        return len(ids)
    def replace_source(self,source_id:str,documents:list[SearchDocument])->None:
        old=self.ids_by_source(source_id);new={d.id for d in documents}
        # Upsert first so readers never see the source disappear entirely.
        self.add_many(documents)
        stale=old-new
        if stale:self.client.delete(collection_name=self.collection_name,points_selector=[self._point_id(i) for i in stale],wait=self.wait)
    def _query(self,query,using:str,top_k:int,filters=None,predicate=None):
        limit=max(top_k,top_k*self.overfetch if predicate else top_k)
        response=self.client.query_points(collection_name=self.collection_name,query=query,using=using,query_filter=self._filter(filters),limit=limit,with_payload=True,with_vectors=False)
        out=[]
        for point in response.points:
            payload=dict(point.payload or {});raw=payload.get("document");data=json.loads(raw) if isinstance(raw,str) else dict(raw or {});data={k:v for k,v in data.items() if k in SearchDocument.__dataclass_fields__};doc=SearchDocument(**data)
            if predicate and not predicate(doc):continue
            out.append((doc.id,float(point.score)))
            if len(out)>=top_k:break
        return out
    def search_dense_dimension(self,dimension,query_vector,top_k,filters=None,predicate=None):return self._query(query_vector,dimension,top_k,filters,predicate)
    def search_sparse(self,query,top_k,filters=None,predicate=None):
        if self.sparse_encoder is None:return []
        sv=self.sparse_encoder.encode_query(query);q=self.models.SparseVector(indices=list(sv),values=[sv[i] for i in sv]);return self._query(q,"sparse",top_k,filters,predicate)
    def search_lexical(self,query,top_k,filters=None,predicate=None):return []
    def search_exact(self,query,top_k,filters=None,predicate=None):
        refs=[DEFAULT_NORMALIZER.normalize_for_search(x) for x in _REFERENCE_RE.findall(query)]
        if not refs:return []
        m=self.models
        if hasattr(m,"MatchAny"):match=m.MatchAny(any=refs)
        else:match=m.MatchValue(value=refs[0])
        qf=self._filter(filters,extra_must=[m.FieldCondition(key="references_normalized",match=match)])
        points,_=self.client.scroll(collection_name=self.collection_name,scroll_filter=qf,limit=max(top_k,self.overfetch*top_k if predicate else top_k),with_payload=True,with_vectors=False)
        out=[]
        for point in points:
            doc=self._from_point(point)
            if predicate and not predicate(doc):continue
            out.append((doc.id,1.0))
            if len(out)>=top_k:break
        return out
