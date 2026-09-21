from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable,Iterable

from ..context import ContextAssembler,ContextPolicy
from ..diversity import mmr_select
from ..feedback import FeedbackStore
from ..models import RetrievalConfig,RetrievalWeights,SearchDocument,SearchHit,SearchResult
from ..routing import HeuristicQueryDecomposer,QueryDecomposer,QueryRouter
from ..security import acl_allows
from ..tracing import SearchTrace
from .enrichment import QueryEnricher
from .fusion import reciprocal_rank_fusion
from .index import BaseSemanticIndex
from .reranker import Reranker
from .scoring import dimension_scores,weighted_score

class SemanticSearchEngine:
    def __init__(self,index:BaseSemanticIndex,query_enricher:QueryEnricher,weights:RetrievalWeights|None=None,reranker:Reranker|None=None,retrieval_config:RetrievalConfig|None=None,*,late_interaction_reranker:Reranker|None=None,query_router:QueryRouter|None=None,query_decomposer:QueryDecomposer|None=None,context_assembler:ContextAssembler|None=None,feedback_store:FeedbackStore|None=None,trace_sink=None,visual_retriever=None,score_calibrator=None)->None:
        self.index=index;self.query_enricher=query_enricher;self.weights=weights or RetrievalWeights();self.reranker=reranker;self.late_interaction_reranker=late_interaction_reranker;self.retrieval_config=retrieval_config or RetrievalConfig();self.query_router=query_router or QueryRouter();self.query_decomposer=query_decomposer or HeuristicQueryDecomposer();self.context_assembler=context_assembler;self.feedback_store=feedback_store;self.trace_sink=trace_sink;self.visual_retriever=visual_retriever;self.score_calibrator=score_calibrator
    @staticmethod
    def _normalize_scores(items:list[tuple[str,float]])->dict[str,float]:
        if not items:return {}
        maximum=max(score for _,score in items);return {doc_id:(score/maximum if maximum>0 else 0.0) for doc_id,score in items}
    @staticmethod
    def _policy(hit:SearchHit,feedback_weight:float)->None:
        trust=max(0.0,min(1.0,float(hit.document.metadata.get("source_trust",1.0))))
        hit.policy_multiplier=trust;relevance=max(0.0,min(1.0,float(hit.score)));hit.relevance_score=relevance
        adjusted=max(0.0,min(1.0,relevance+feedback_weight*hit.feedback_score));hit.score=adjusted*trust
        if trust<1.0:hit.reasons.append(f"source_trust={trust:.2f}")
        if hit.feedback_score:hit.reasons.append(f"feedback={hit.feedback_score:.3f}")
    def search(self,query:str,top_k:int=10,minimum_score:float=0.0,filters:dict|None=None,predicate:Callable[[SearchDocument],bool]|None=None,weights:RetrievalWeights|None=None,*,hybrid:bool|None=None,candidate_k:int|None=None,principal:str|None=None,groups:Iterable[str]|None=None,tenant_id:str|None=None,expand_context:bool|None=None)->SearchResult:
        if not query or not query.strip():raise ValueError("query cannot be empty")
        if top_k<1:raise ValueError("top_k must be >= 1")
        if not 0<=minimum_score<=1:raise ValueError("minimum_score must be between 0 and 1")
        clean=query.strip();cfg=self.retrieval_config;use_hybrid=cfg.hybrid if hybrid is None else hybrid;trace=SearchTrace();trace.attributes["query"]=clean
        policy=self.query_router.route(clean) if cfg.use_query_router else self.query_router.route("normal query with enough terms");trace.attributes["retrieval_policy"]=policy.name
        group_set=set(groups or [])
        def secured(doc:SearchDocument)->bool:
            if not acl_allows(doc,principal=principal,groups=group_set,tenant_id=tenant_id,tenant_mode=cfg.tenant_mode):return False
            return predicate(doc) if predicate else True
        active_weights=weights or self.weights;candidate_limit=candidate_k or max(top_k*cfg.candidate_multiplier,cfg.min_candidates);subqueries=self.query_decomposer.decompose(clean) if (cfg.use_multi_query and policy.use_multi_query) else [clean]
        effective_filters=dict(filters or {});effective_filters["__security__"]={"principal":principal,"groups":sorted(group_set),"tenant_id":tenant_id,"tenant_mode":cfg.tenant_mode}
        trace.attributes["security_pushdown"]=True
        rankings=[];ranking_weights=[];dense_raw=defaultdict(float);lexical_raw=defaultdict(float);sparse_raw=defaultdict(float);exact_raw=defaultdict(float);graph_raw=defaultdict(float);visual_raw=defaultdict(float);reasons=defaultdict(list)
        with trace.span("candidate_generation",subqueries=len(subqueries)) as span:
            for subquery in subqueries:
                qv=self.query_enricher.enrich(subquery);dimensions=qv.dimensions() if cfg.use_multi_representation else {"raw_text":qv.raw_text}
                for dim,qvec in dimensions.items():
                    if not qvec:continue
                    hits=self.index.search_dense_dimension(dim,qvec,min(candidate_limit,cfg.dense_per_dimension),filters=effective_filters,predicate=secured)
                    if hits:
                        rankings.append([x for x,_ in hits]);ranking_weights.append(policy.dense_weight*active_weights.as_dict().get(dim,1.0))
                        for doc_id,score in hits:dense_raw[doc_id]=max(dense_raw[doc_id],score);reasons[doc_id].append(f"dense:{dim}")
                if use_hybrid:
                    for name,items,weight,rawmap in (
                        ("bm25",self.index.search_lexical(subquery,min(candidate_limit,cfg.lexical_top_k),filters=effective_filters,predicate=secured),policy.lexical_weight,lexical_raw),
                        ("sparse",self.index.search_sparse(subquery,min(candidate_limit,cfg.sparse_top_k),filters=effective_filters,predicate=secured),policy.sparse_weight,sparse_raw),
                        ("exact",self.index.search_exact(subquery,min(candidate_limit,cfg.exact_top_k),filters=effective_filters,predicate=secured),policy.exact_weight,exact_raw)):
                        if items:
                            rankings.append([x for x,_ in items]);ranking_weights.append(weight)
                            for doc_id,score in items:rawmap[doc_id]=max(rawmap[doc_id],score);reasons[doc_id].append(name)
                    if policy.use_graph:
                        mode="global" if policy.name=="global" else ("local" if policy.name=="entity-relational" else "drift")
                        try:graph=self.index.search_graph(subquery,candidate_limit,filters=effective_filters,predicate=secured,mode=mode)
                        except Exception as exc:
                            if not cfg.fail_open_optional_stages:raise
                            graph=[];trace.attributes.setdefault("optional_stage_errors",[]).append(f"graph:{type(exc).__name__}")
                        if graph:
                            rankings.append([x for x,_ in graph]);ranking_weights.append(.7)
                            for doc_id,score in graph:graph_raw[doc_id]=max(graph_raw[doc_id],score);reasons[doc_id].append("graph");reasons[doc_id].append(f"graph_mode:{mode}")
                    if self.visual_retriever is not None:
                        visual=[]
                        try:
                            visual_rows=self.visual_retriever.search(subquery,top_k=min(candidate_limit,cfg.visual_top_k))
                        except Exception as exc:
                            if not cfg.fail_open_optional_stages:raise
                            visual_rows=[];trace.attributes.setdefault("optional_stage_errors",[]).append(f"visual:{type(exc).__name__}")
                        for item,score in visual_rows:
                            meta=getattr(item,"metadata",{}) or {};doc_id=str(meta.get("document_id") or meta.get("chunk_id") or getattr(item,"id",""))
                            doc=self.index.get(doc_id) if doc_id else None
                            if doc is not None and secured(doc):visual.append((doc_id,float(score)))
                        if visual:
                            rankings.append([x for x,_ in visual]);ranking_weights.append(cfg.visual_weight)
                            for doc_id,score in visual:visual_raw[doc_id]=max(visual_raw[doc_id],score);reasons[doc_id].append("visual")
            span["rankings"]=len(rankings)
        fusion=reciprocal_rank_fusion(rankings,k=cfg.rrf_k,weights=ranking_weights) if rankings else {}
        # No retrieval evidence means no result. Never manufacture arbitrary candidates.
        ordered_ids=sorted(fusion,key=fusion.get,reverse=True)[:max(candidate_limit,cfg.fusion_top_k)] if fusion else []
        qv=self.query_enricher.enrich(clean) if ordered_ids else None
        dense_norm=self._normalize_scores(list(dense_raw.items()));lexical_norm=self._normalize_scores(list(lexical_raw.items()));sparse_norm=self._normalize_scores(list(sparse_raw.items()));exact_norm=self._normalize_scores(list(exact_raw.items()));graph_norm=self._normalize_scores(list(graph_raw.items()));visual_norm=self._normalize_scores(list(visual_raw.items()))
        hits=[];candidate_docs=self.index.get_many(ordered_ids)
        with trace.span("multi_vector_scoring",candidates=len(ordered_ids)):
            for doc_id in ordered_ids:
                doc=candidate_docs.get(doc_id)
                if doc is None or doc.vectors is None:continue
                if qv is None:continue
                scores=dimension_scores(qv,doc.vectors);semantic=weighted_score(scores,active_weights,qv,doc.vectors);fused=fusion.get(doc_id,0.0)
                if use_hybrid:
                    evidence=max(semantic,exact_norm.get(doc_id,0.0),.92*sparse_norm.get(doc_id,0.0),.85*lexical_norm.get(doc_id,0.0),.80*graph_norm.get(doc_id,0.0),.85*visual_norm.get(doc_id,0.0));final=cfg.semantic_blend*evidence+(1-cfg.semantic_blend)*fused
                else:final=semantic
                hit=SearchHit(document=doc,score=final,dimension_scores=scores,dense_score=dense_norm.get(doc_id,0),lexical_score=lexical_norm.get(doc_id,0),sparse_score=sparse_norm.get(doc_id,0),exact_score=exact_norm.get(doc_id,0),graph_score=graph_norm.get(doc_id,0),visual_score=visual_norm.get(doc_id,0),fusion_score=fused,semantic_score=semantic,relevance_score=final,reasons=list(dict.fromkeys(reasons.get(doc_id,[]))))
                if self.feedback_store is not None:hit.feedback_score=self.feedback_store.score(clean,doc_id)
                hits.append(hit)
        hits.sort(key=lambda h:h.score,reverse=True)
        if self.late_interaction_reranker and hits:
            with trace.span("late_interaction",candidates=min(len(hits),cfg.late_interaction_top_k)):
                before=list(hits);scores=[h.score for h in hits]
                try:hits=self.late_interaction_reranker.rerank(clean,hits[:cfg.late_interaction_top_k],cfg.late_interaction_top_k)+hits[cfg.late_interaction_top_k:]
                except Exception as exc:
                    if not cfg.fail_open_optional_stages:raise
                    hits=before
                    for h,score in zip(hits,scores):h.score=score
                    trace.attributes.setdefault("optional_stage_errors",[]).append(f"late_interaction:{type(exc).__name__}")
        if self.reranker and hits:
            with trace.span("cross_encoder",candidates=min(len(hits),cfg.rerank_top_k)):
                before=list(hits);scores=[h.score for h in hits]
                try:hits=self.reranker.rerank(clean,hits[:cfg.rerank_top_k],cfg.rerank_top_k)+hits[cfg.rerank_top_k:]
                except Exception as exc:
                    if not cfg.fail_open_optional_stages:raise
                    hits=before
                    for h,score in zip(hits,scores):h.score=score
                    trace.attributes.setdefault("optional_stage_errors",[]).append(f"reranker:{type(exc).__name__}")
        # Hard policy/trust is deliberately applied after every learned or callable ranker.
        for hit in hits:
            self._policy(hit,cfg.feedback_weight)
            if self.score_calibrator is not None:
                try:hit.calibrated_score=float(self.score_calibrator(hit.score))
                except Exception as exc:
                    if not cfg.fail_open_optional_stages:raise
                    trace.attributes.setdefault("optional_stage_errors",[]).append(f"calibrator:{type(exc).__name__}")
        hits=[h for h in hits if h.score>=minimum_score];hits.sort(key=lambda h:h.score,reverse=True)
        with trace.span("diversity"):final_hits=mmr_select(hits,top_k,cfg.mmr_lambda,cfg.max_per_source) if cfg.use_diversity else hits[:top_k]
        context_docs=[];should_expand=cfg.auto_expand_context if expand_context is None else expand_context
        if should_expand and self.context_assembler is not None:
            with trace.span("context_assembly",budget=cfg.context_token_budget):context_docs=self.context_assembler.assemble(clean,final_hits,ContextPolicy(cfg.context_token_budget,cfg.context_neighbors,True,True))
        trace.attributes.update({"dense_candidates":len(dense_raw),"lexical_candidates":len(lexical_raw),"sparse_candidates":len(sparse_raw),"exact_candidates":len(exact_raw),"visual_candidates":len(visual_raw),"fused_candidates":len(fusion),"final_hits":len(final_hits)})
        if self.trace_sink is not None:
            try:self.trace_sink(trace)
            except Exception as exc:
                if not cfg.fail_open_optional_stages:raise
                trace.attributes.setdefault("optional_stage_errors",[]).append(f"trace_sink:{type(exc).__name__}")
        return SearchResult(clean,final_hits,len(hits),"hybrid" if use_hybrid else "dense",{**trace.attributes,"trace":trace.as_dict(),"candidate_ids":ordered_ids,"score_semantics":"ranking_score_not_probability"},context_docs)
