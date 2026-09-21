from __future__ import annotations

from ..models import SearchDocument
from .index import BaseSemanticIndex

class CallableSemanticIndex(BaseSemanticIndex):
    """SDK-agnostic production adapter.

    Backends may implement source-scoped operations directly so the facade never needs
    to enumerate a remote collection. Search callbacks should enforce filters server-side;
    the optional predicate is a final defense-in-depth check when returned documents exist.
    """
    def __init__(self,*,add_fn,remove_fn,get_fn,dense_fn,documents_fn=None,add_many_fn=None,remove_source_fn=None,replace_source_fn=None,documents_by_source_fn=None,lexical_fn=None,sparse_fn=None,exact_fn=None,graph_fn=None,clear_fn=None)->None:
        self.add_fn=add_fn;self.add_many_fn=add_many_fn;self.remove_fn=remove_fn;self.remove_source_fn=remove_source_fn;self.replace_source_fn=replace_source_fn;self.get_fn=get_fn;self.documents_fn=documents_fn;self.documents_by_source_fn=documents_by_source_fn;self.dense_fn=dense_fn;self.lexical_fn=lexical_fn;self.sparse_fn=sparse_fn;self.exact_fn=exact_fn;self.graph_fn=graph_fn;self.clear_fn=clear_fn
    def add(self,document:SearchDocument)->None:self.add_fn(document)
    def add_many(self,documents:list[SearchDocument])->None:
        if self.add_many_fn:self.add_many_fn(documents)
        else:super().add_many(documents)
    def remove(self,document_id:str)->bool:return bool(self.remove_fn(document_id))
    def remove_source(self,source_id:str)->int:
        if self.remove_source_fn:return int(self.remove_source_fn(source_id))
        return super().remove_source(source_id)
    def replace_source(self,source_id:str,documents:list[SearchDocument])->None:
        if self.replace_source_fn:self.replace_source_fn(source_id,documents)
        else:super().replace_source(source_id,documents)
    def clear(self)->None:
        if self.clear_fn:self.clear_fn()
    def get(self,document_id:str)->SearchDocument|None:return self.get_fn(document_id)
    def documents(self,filters=None,predicate=None):
        if self.documents_fn is None:raise NotImplementedError("remote backend did not provide documents_fn; use source-scoped operations")
        docs=list(self.documents_fn(filters));return [d for d in docs if predicate(d)] if predicate else docs
    def documents_by_source(self,source_id:str)->list[SearchDocument]:
        if self.documents_by_source_fn:return list(self.documents_by_source_fn(source_id))
        return super().documents_by_source(source_id)
    def search_dense_dimension(self,dimension,query_vector,top_k,filters=None,predicate=None):return self.dense_fn(dimension,query_vector,top_k,filters,predicate)
    def search_lexical(self,query,top_k,filters=None,predicate=None):return self.lexical_fn(query,top_k,filters,predicate) if self.lexical_fn else []
    def search_sparse(self,query,top_k,filters=None,predicate=None):return self.sparse_fn(query,top_k,filters,predicate) if self.sparse_fn else []
    def search_exact(self,query,top_k,filters=None,predicate=None):return self.exact_fn(query,top_k,filters,predicate) if self.exact_fn else []
    def search_graph(self,query,top_k,filters=None,predicate=None,mode="drift"):return self.graph_fn(query,top_k,filters,predicate,mode) if self.graph_fn else []
