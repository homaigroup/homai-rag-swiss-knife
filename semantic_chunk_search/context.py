from __future__ import annotations

from dataclasses import dataclass
import re

from .models import ParentDocument,SearchDocument,SearchHit
from .tokenization import ApproximateTokenCounter,BaseTokenCounter,truncate_tokens

_IDENTIFIER_RE=re.compile(r"(?:https?://\S+|\b[A-Z]{2,}[\-_]?\d{2,}\b|\b\w+[\-_]\d{3,}\b)",re.I)

@dataclass(slots=True)
class ContextPolicy:
    token_budget:int=4000
    neighbors:int=1
    include_parent:bool=True
    query_aware:bool=True
    max_sources:int|None=None

class ContextAssembler:
    def __init__(self,index,parents:dict[str,ParentDocument]|None=None,token_counter:BaseTokenCounter|None=None)->None:
        self.index=index;self.parents=parents if parents is not None else {};self.counter=token_counter or ApproximateTokenCounter()

    def _append(self,out:list[SearchDocument],seen:set[str],candidate:SearchDocument,remaining:int,*,must_include:bool=False)->int:
        if candidate.id in seen:return remaining
        cost=self.counter.count(candidate.text)
        if cost<=remaining:
            out.append(candidate);seen.add(candidate.id);return remaining-cost
        if remaining<=0:return remaining
        # Mandatory hits are clipped rather than dropped; optional context needs a useful minimum budget.
        if must_include or remaining>=32:
            clipped=SearchDocument(id=candidate.id+":truncated",text=truncate_tokens(candidate.text,remaining,self.counter),source_id=candidate.source_id,metadata={**candidate.metadata,"truncated":True},section_path=list(candidate.section_path))
            if clipped.text:out.append(clipped);seen.add(candidate.id);return 0
        return remaining

    def assemble(self,query:str,hits:list[SearchHit],policy:ContextPolicy|None=None)->list[SearchDocument]:
        policy=policy or ContextPolicy();remaining=policy.token_budget;out=[];seen:set[str]=set();sources:set[str]=set();exact=bool(_IDENTIFIER_RE.search(query)) if policy.query_aware else False
        for hit in hits:
            if remaining<=0:break
            doc=hit.document;source=str(doc.source_id or "")
            if policy.max_sources is not None and source not in sources and len(sources)>=policy.max_sources:continue
            # Invariant: a retrieved hit is never displaced by its parent/neighbor context.
            hit_doc=SearchDocument(**{name:getattr(doc,name) for name in SearchDocument.__dataclass_fields__})
            hit_doc.metadata={**doc.metadata,"context_role":"hit"}
            remaining=self._append(out,seen,hit_doc,remaining,must_include=True);sources.add(source)
            if remaining<=0:break
            if exact:continue
            if policy.include_parent and doc.parent_id and doc.parent_id in self.parents:
                parent=self.parents[doc.parent_id]
                pdoc=SearchDocument(id=parent.id,text=parent.text,source_id=parent.source_id,metadata={**parent.metadata,"context_role":"parent"},section_path=parent.section_path)
                remaining=self._append(out,seen,pdoc,remaining)
                if remaining<=0:break
            window=getattr(self.index,"context_window",lambda _id,neighbors=1:[doc])(doc.id,neighbors=policy.neighbors) or [doc]
            for candidate in window:
                if candidate.id==doc.id:continue
                clone=SearchDocument(**{name:getattr(candidate,name) for name in SearchDocument.__dataclass_fields__});clone.metadata={**candidate.metadata,"context_role":"neighbor"}
                remaining=self._append(out,seen,clone,remaining)
                if remaining<=0:break
        return out
