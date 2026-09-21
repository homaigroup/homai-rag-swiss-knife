from __future__ import annotations

import asyncio
from dataclasses import asdict,replace
from datetime import datetime,timezone
import hashlib
import json
from threading import RLock

from .advanced import SimpleEntityGraph
from .feedback import FeedbackStore
from .chunking import ChunkingFactory
from .context import ContextAssembler
from .embedding import BaseEmbeddingProvider,HashEmbeddingProvider
from .models import ChunkingConfig,ParentDocument,RetrievalConfig,SearchDocument
from .normalization import DEFAULT_NORMALIZER,NormalizedText,TextNormalizer
from .persistence import SQLiteDocumentStore
from .routing import QueryDecomposer,QueryRouter
from .security import ContentSecurityScanner
from .sparse import BaseSparseEncoder
from .tokenization import ApproximateTokenCounter,BaseTokenCounter,TokenizerObjectCounter
from .search import BaseSemanticIndex,DocumentEnricher,DocumentVectorizer,HNSWSemanticIndex,InMemorySemanticIndex,QueryEnricher,SemanticSearchEngine
from .search.reranker import Reranker

class SemanticChunkSearch:
    """Hardened facade: deterministic ingestion, ACL-safe updates, durable persistence and hybrid retrieval."""
    def __init__(self,chunking_config:ChunkingConfig|None=None,embedder:BaseEmbeddingProvider|None=None,*,index:BaseSemanticIndex|None=None,use_ann:bool=False,retrieval_config:RetrievalConfig|None=None,reranker:Reranker|None=None,late_interaction_reranker:Reranker|None=None,sparse_encoder:BaseSparseEncoder|None=None,query_llm=None,contextualizer=None,query_router:QueryRouter|None=None,query_decomposer:QueryDecomposer|None=None,token_counter:BaseTokenCounter|None=None,normalizer:TextNormalizer|None=None,persistence_path:str|None=None,security_scanner:ContentSecurityScanner|None=None,enable_graph:bool=True,feedback_store:FeedbackStore|None=None,enforce_embedding_token_budget:bool=True,trace_sink=None,visual_retriever=None,score_calibrator=None)->None:
        self._lock=RLock();self.normalizer=normalizer or DEFAULT_NORMALIZER;self.embedder=embedder or HashEmbeddingProvider(normalizer=self.normalizer)
        if token_counter is None and getattr(getattr(self.embedder,"model",None),"tokenizer",None) is not None:
            token_counter=TokenizerObjectCounter(self.embedder.model.tokenizer,self.embedder.model_name)
        self.token_counter=token_counter or ApproximateTokenCounter(self.normalizer)
        cfg=chunking_config or ChunkingConfig()
        max_input=getattr(self.embedder,"max_input_tokens",None)
        if enforce_embedding_token_budget and max_input and cfg.max_chunk_tokens>max(32,max_input-16):
            effective=max(32,max_input-16);cfg=replace(cfg,max_chunk_tokens=effective,min_chunk_tokens=min(cfg.min_chunk_tokens,effective),overlap_tokens=min(cfg.overlap_tokens,max(0,effective-1)))
        self.chunking_config=cfg;self.chunker=ChunkingFactory(cfg,self.embedder,self.token_counter);self.security_scanner=security_scanner or ContentSecurityScanner();self.feedback_store=feedback_store or FeedbackStore();self.parents:dict[str,ParentDocument]={}
        self._chunking_signature=self._sig({"config":asdict(cfg),"tokenizer":self.token_counter.signature,"normalizer":self.normalizer.signature})
        self.document_enricher=DocumentEnricher(self.normalizer,contextualizer=contextualizer);self.vectorizer=DocumentVectorizer(self.embedder,self.document_enricher);graph=SimpleEntityGraph() if enable_graph else None
        if index is not None:self.index=index
        elif use_ann:self.index=HNSWSemanticIndex(self.vectorizer,dimensions=self.embedder.dimensions,sparse_encoder=sparse_encoder,graph=graph)
        else:self.index=InMemorySemanticIndex(self.vectorizer,sparse_encoder=sparse_encoder,graph=graph)
        self.context_assembler=ContextAssembler(self.index,self.parents,self.token_counter)
        self.search_engine=SemanticSearchEngine(self.index,QueryEnricher(self.embedder,llm=query_llm,normalizer=self.normalizer),reranker=reranker,retrieval_config=retrieval_config,late_interaction_reranker=late_interaction_reranker,query_router=query_router,query_decomposer=query_decomposer,context_assembler=self.context_assembler,feedback_store=self.feedback_store,trace_sink=trace_sink,visual_retriever=visual_retriever,score_calibrator=score_calibrator)
        self.store=SQLiteDocumentStore(persistence_path) if persistence_path else None
        self.compatibility={"reembedded":0,"legacy_chunking":0}
        if self.store is not None:
            persisted=self.store.load_all();dirty=False
            for doc in persisted:
                incompatible=(bool(doc.embedding_signature and doc.embedding_signature!=self.embedder.signature) or bool(doc.embedding_model and (doc.embedding_model!=self.embedder.model_name or (doc.embedding_dimensions and doc.embedding_dimensions!=self.embedder.dimensions))))
                if incompatible:doc.vectors=None;self.compatibility["reembedded"]+=1;dirty=True
                if doc.chunking_signature and doc.chunking_signature!=self._chunking_signature:self.compatibility["legacy_chunking"]+=1
                doc.embedding_model=self.embedder.model_name;doc.embedding_dimensions=self.embedder.dimensions;doc.embedding_signature=self.embedder.signature
                current_retrieval=self._retrieval_signature(doc.text,doc.metadata,doc.section_path)
                if doc.retrieval_signature!=current_retrieval:
                    doc.retrieval_signature=current_retrieval
                    if self.document_enricher.signature not in ("",):
                        # Re-enrichment/vectorization is required when contextualization contract changed.
                        doc.vectors=None;doc.context_text="";doc.summary="";doc.keywords=[];doc.references=[];dirty=True
            if persisted:
                self.index.add_many(persisted);self._rebuild_parents(persisted)
                if dirty:self.store.upsert_many(persisted)
            self.store.set_metadata("active_embedding_signature",self.embedder.signature);self.store.set_metadata("active_chunking_signature",self._chunking_signature);self.store.set_metadata("active_vectorizer_signature",self.vectorizer.signature)
            for q,d,r,k in self.store.load_feedback():self.feedback_store.record(q,d,r,k)

    @staticmethod
    def _hash(text:str)->str:return hashlib.sha256(text.encode("utf-8")).hexdigest()
    @staticmethod
    def _sig(value)->str:return hashlib.blake2b(json.dumps(value,sort_keys=True,ensure_ascii=False,default=str).encode(),digest_size=12).hexdigest()
    def _retrieval_signature(self,text:str,meta:dict,section_path:list[str])->str:
        semantic_meta={k:meta.get(k) for k in ("title","document_title","parent_context","language","content_type") if meta.get(k) is not None}
        return self._sig({"text":text,"section":section_path,"meta":semantic_meta,"vectorizer":self.vectorizer.signature,"enrichment":"4"})

    def _rebuild_parents(self,docs:list[SearchDocument])->None:
        grouped:dict[str,list[SearchDocument]]={}
        for doc in docs:
            if doc.parent_id:grouped.setdefault(doc.parent_id,[]).append(doc)
        for pid,children in grouped.items():
            children.sort(key=lambda d:int(d.chunk_index or 0));source=children[0].source_id or "";section=list(children[0].section_path)
            candidates=[str(d.metadata.get("parent_text") or "").strip() for d in children if str(d.metadata.get("parent_text") or "").strip()]
            parent_text=max(candidates,key=len) if candidates else "\n\n".join(d.text for d in children)
            meta={k:v for k,v in children[0].metadata.items() if k!="parent_text"};self.parents[pid]=ParentDocument(pid,source,parent_text,section,meta,[d.id for d in children],self.token_counter.count(parent_text))

    def _document_from_chunk(self,chunk,version:str,now:str,norm:NormalizedText,page_no:int|None=None)->SearchDocument:
        section_path=list(chunk.metadata.get("section_path",[]));parent_id=chunk.parent_id or chunk.metadata.get("parent_id");ns,ne=chunk.start_char,chunk.end_char;os,oe=norm.original_span(ns,ne)
        meta={**chunk.metadata,"chunk_type":chunk.chunk_type.value};sig=self._retrieval_signature(chunk.raw_text,meta,section_path)
        return SearchDocument(id=chunk.id,text=chunk.raw_text,source_id=chunk.source_id,chunk_index=chunk.chunk_index,metadata=meta,parent_id=str(parent_id) if parent_id else None,section_path=section_path,titles=section_path.copy(),source_version=version,content_hash=self._hash(chunk.raw_text),embedding_model=self.embedder.model_name,embedding_dimensions=self.embedder.dimensions,embedding_signature=self.embedder.signature,chunking_signature=self._chunking_signature,retrieval_signature=sig,created_at=now,start_char=os,end_char=oe,normalized_start_char=ns,normalized_end_char=ne,page_start=page_no or chunk.metadata.get("page_start"),page_end=page_no or chunk.metadata.get("page_end"))

    def _prepare_reuse(self,docs:list[SearchDocument],existing:dict[str,SearchDocument])->None:
        for doc in docs:
            old=existing.get(doc.id)
            if old and old.retrieval_signature==doc.retrieval_signature and old.embedding_signature==self.embedder.signature and old.vectors is not None:
                doc.vectors=old.vectors;doc.summary=old.summary;doc.keywords=list(old.keywords);doc.titles=list(old.titles) if old.titles else list(doc.titles);doc.references=list(old.references);doc.context_text=old.context_text

    def _commit_source(self,source_id:str,docs:list[SearchDocument])->list[SearchDocument]:
        old=self.index.documents_by_source(source_id);old_map={d.id:d for d in old};self._prepare_reuse(docs,old_map)
        try:
            self.index.replace_source(source_id,docs)
            canonical=[self.index.get(d.id) or d for d in docs]
            if self.store is not None:self.store.replace_source(source_id,canonical)
        except Exception as exc:
            try:self.index.restore_source(source_id,old)
            except Exception as rollback_exc:raise RuntimeError("source update failed and index rollback also failed") from exc
            raise
        self._drop_source_parents(source_id);self._rebuild_parents(canonical);return canonical

    def ingest(self,text:str,source_id:str,strategy:str|None=None,metadata:dict|None=None,*,source_version:str|None=None,allow_quarantined:bool=False)->list[SearchDocument]:
        if not source_id:raise ValueError("source_id cannot be empty")
        norm=self.normalizer.normalize_with_offsets(text)
        if not norm.text:raise ValueError("text cannot be empty")
        report=self.security_scanner.scan(norm.text)
        if not report.allowed and not allow_quarantined:raise ValueError("content quarantined: "+", ".join(f.kind for f in report.findings))
        base_meta=dict(metadata or {});version=str(source_version or base_meta.get("source_version") or "1");base_meta["source_version"]=version;base_meta["source_content_hash"]=self._hash(norm.text);base_meta["original_content_hash"]=self._hash(text)
        if report.findings:base_meta["security_findings"]=[{"kind":f.kind,"severity":f.severity,"detail":f.detail} for f in report.findings]
        now=datetime.now(timezone.utc).isoformat();chunks=self.chunker.chunk(norm.text,source_id,strategy=strategy,metadata=base_meta);docs=[self._document_from_chunk(c,version,now,norm) for c in chunks]
        for i,d in enumerate(docs):d.previous_id=docs[i-1].id if i else None;d.next_id=docs[i+1].id if i+1<len(docs) else None
        with self._lock:return self._commit_source(source_id,docs)

    def ingest_many(self,items:list[tuple[str,str,dict|None]])->list[SearchDocument]:
        out=[]
        for text,source_id,metadata in items:out.extend(self.ingest(text,source_id,metadata=metadata))
        return out
    async def aingest(self,*args,**kwargs)->list[SearchDocument]:return await asyncio.to_thread(self.ingest,*args,**kwargs)
    async def asearch(self,*args,**kwargs):return await asyncio.to_thread(self.search,*args,**kwargs)
    def _drop_source_parents(self,source_id:str)->None:
        for pid,parent in list(self.parents.items()):
            if str(parent.source_id)==str(source_id):self.parents.pop(pid,None)
    def delete_source(self,source_id:str)->int:
        with self._lock:
            old_docs=self.index.documents_by_source(source_id);ids={d.id for d in old_docs}
            if not old_docs:return 0
            # Persist the privacy-sensitive deletion atomically first; if the in-memory
            # backend unexpectedly fails afterwards, restore the durable snapshot.
            if self.store is not None:
                persisted_old=self.store.load_source(source_id)
                feedback_old=[row for row in self.store.load_feedback() if row[1] in ids]
                self.store.delete_source_cascade(source_id)
            else:
                persisted_old=[];feedback_old=[]
            try:
                count=self.index.remove_source(source_id)
            except Exception as exc:
                rollback_errors=[]
                try:self.index.restore_source(source_id,old_docs)
                except Exception as rollback_exc:rollback_errors.append(rollback_exc)
                if self.store is not None:
                    try:
                        self.store.replace_source(source_id,persisted_old)
                        for q,d,r,k in feedback_old:self.store.append_feedback(q,d,r,k)
                    except Exception as rollback_exc:rollback_errors.append(rollback_exc)
                if rollback_errors:raise RuntimeError("source deletion failed and rollback was incomplete") from exc
                raise
            self._drop_source_parents(source_id);self.feedback_store.remove_documents(ids)
            clear_cache=getattr(self.embedder,"clear_cache",None)
            if clear_cache:clear_cache()
            return count
    def search(self,query:str,**kwargs):return self.search_engine.search(query,**kwargs)
    def context_window(self,document_id:str,neighbors:int=1)->list[SearchDocument]:
        fn=getattr(self.index,"context_window",None);doc=self.index.get(document_id);return fn(document_id,neighbors=neighbors) if fn else ([doc] if doc else [])
    def record_feedback(self,query:str,document_id:str,relevance:float,kind:str="explicit")->None:
        q=self.normalizer.normalize_for_search(query);rel=max(-1.0,min(1.0,float(relevance)))
        # Durable-first avoids a session-only feedback state when persistence fails.
        if self.store is not None:self.store.append_feedback(q,document_id,rel,kind)
        self.feedback_store.record(q,document_id,rel,kind)

    def ingest_pages(self,pages:list[str],source_id:str,strategy:str|None=None,metadata:dict|None=None,*,source_version:str|None=None,allow_quarantined:bool=False,page_metadata:dict[int,dict]|None=None)->list[SearchDocument]:
        if not pages:return []
        if not source_id:raise ValueError("source_id cannot be empty")
        base_meta=dict(metadata or {});version=str(source_version or base_meta.get("source_version") or "1");now=datetime.now(timezone.utc).isoformat();docs=[];page_metadata=page_metadata or {}
        prepared=[];all_findings=[]
        for page_no,page in enumerate(pages,start=1):
            norm=self.normalizer.normalize_with_offsets(page or "")
            if not norm.text:continue
            report=self.security_scanner.scan(norm.text)
            if not report.allowed and not allow_quarantined:raise ValueError(f"page {page_no} content quarantined: "+", ".join(f.kind for f in report.findings))
            findings=[{"page":page_no,"kind":f.kind,"severity":f.severity,"detail":f.detail} for f in report.findings];all_findings.extend(findings);prepared.append((page_no,page,norm,findings))
        if not prepared:return []
        normalized_full="\n\f\n".join(norm.text for _,_,norm,_ in prepared);original_full="\n\f\n".join(str(page or "") for page in pages)

        source_hash=self._hash(normalized_full);original_hash=self._hash(original_full)
        for page_no,_page,norm,findings in prepared:
            page_meta={**base_meta,**dict(page_metadata.get(page_no,{}) or {}),"source_version":version,"page_start":page_no,"page_end":page_no,"source_content_hash":source_hash,"original_content_hash":original_hash}
            if findings:page_meta["security_findings"]=findings
            for chunk in self.chunker.chunk(norm.text,source_id,strategy=strategy,metadata=page_meta):
                chunk.id=f"{chunk.id}:p{page_no}";doc=self._document_from_chunk(chunk,version,now,norm,page_no);doc.chunk_index=len(docs);docs.append(doc)
        for i,d in enumerate(docs):d.previous_id=docs[i-1].id if i else None;d.next_id=docs[i+1].id if i+1<len(docs) else None
        with self._lock:return self._commit_source(source_id,docs)

    def ingest_document(self,path,source_id:str|None=None,loader=None,**kwargs):
        from pathlib import Path
        from .ingestion import loader_for_path
        selected=loader or loader_for_path(path);loaded=selected.load(path);sid=source_id or str(Path(path).resolve())
        metadata={**loaded.metadata,**dict(kwargs.pop("metadata",{}) or {})}
        # Preserve layout/page provenance supplied by richer custom loaders without
        # coupling the core to any specific OCR/layout vendor.
        by_page={}
        for block in loaded.blocks:
            page=int(block.page or 1);entry={"block_type":block.block_type,"bbox":list(block.bbox) if block.bbox is not None else None,"metadata":dict(block.metadata or {})}
            if block.text:entry["text"]=block.text
            by_page.setdefault(page,[]).append(entry)
        page_meta={page:{"layout_blocks":blocks} for page,blocks in by_page.items()}
        return self.ingest_pages(loaded.pages,sid,metadata=metadata,page_metadata=page_meta,**kwargs)

    def close(self)->None:
        if self.store is not None:self.store.close()
