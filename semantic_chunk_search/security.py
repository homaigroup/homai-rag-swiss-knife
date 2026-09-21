from __future__ import annotations

from dataclasses import dataclass,field
import base64
import html
import re
from typing import Iterable

from .models import SearchDocument

_PROMPT_PATTERNS=[
    re.compile(r"ignore\s+(?:all|any|the)?\s*(?:previous|prior)\s+instructions",re.I),
    re.compile(r"system\s+prompt",re.I),re.compile(r"developer\s+message",re.I),
    re.compile(r"do\s+not\s+follow\s+.*instructions",re.I),re.compile(r"jailbreak",re.I),
    re.compile(r"نادیده\s+بگیر.*(?:دستور|راهنما)",re.I),re.compile(r"پرامپت\s+سیستم",re.I),
]
_REPEAT_RE=re.compile(r"\b([\w\-]{3,})\b(?:\s+\1\b){6,}",re.I)
_HIDDEN_HTML_RE=re.compile(r"(?:display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0|color\s*:\s*white)",re.I)

@dataclass(slots=True)
class SecurityFinding:
    kind:str;severity:str;detail:str
@dataclass(slots=True)
class SecurityReport:
    allowed:bool;findings:list[SecurityFinding]=field(default_factory=list)

class ContentSecurityScanner:
    """Defense-in-depth heuristic scanner; access control remains the hard security boundary."""
    def __init__(self,quarantine_on_prompt_injection:bool=False,quarantine_on_keyword_stuffing:bool=False,quarantine_on_hidden_content:bool=False)->None:
        self.quarantine_on_prompt_injection=quarantine_on_prompt_injection;self.quarantine_on_keyword_stuffing=quarantine_on_keyword_stuffing;self.quarantine_on_hidden_content=quarantine_on_hidden_content
    def scan(self,text:str)->SecurityReport:
        findings=[];decoded=html.unescape(text).replace("\u200b","").replace("\u200c"," ").replace("\u200d","").replace("\ufeff","")
        decoded_candidates=[decoded]
        for token in re.findall(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/=])",decoded)[:8]:
            try:
                raw=base64.b64decode(token,validate=True);candidate=raw.decode("utf-8")
                if candidate and sum(ch.isprintable() for ch in candidate)/len(candidate)>.9:decoded_candidates.append(candidate)
            except Exception:pass
        scan_text="\n".join(decoded_candidates)
        for pattern in _PROMPT_PATTERNS:
            if pattern.search(scan_text):findings.append(SecurityFinding("prompt_injection","high",pattern.pattern));break
        if _REPEAT_RE.search(decoded):findings.append(SecurityFinding("keyword_stuffing","medium","repeated token sequence"))
        if _HIDDEN_HTML_RE.search(decoded):findings.append(SecurityFinding("hidden_content","medium","hidden/zero-visibility HTML style"))
        blocked=any((f.kind=="prompt_injection" and self.quarantine_on_prompt_injection) or (f.kind=="keyword_stuffing" and self.quarantine_on_keyword_stuffing) or (f.kind=="hidden_content" and self.quarantine_on_hidden_content) for f in findings)
        return SecurityReport(not blocked,findings)

def acl_allows(document:SearchDocument,principal:str|None=None,groups:Iterable[str]|None=None,tenant_id:str|None=None,tenant_mode:str="isolated")->bool:
    meta=document.metadata;doc_tenant=meta.get("tenant_id")
    if tenant_mode=="isolated":
        # Tenant-scoped documents are never globally visible, even if marked public.
        if doc_tenant is not None and str(doc_tenant)!=str(tenant_id):return False
    elif tenant_mode=="shared_public":
        if tenant_id is not None and doc_tenant not in (None,tenant_id):return False
    elif tenant_mode=="global":pass
    else:raise ValueError("unknown tenant_mode")
    visibility=meta.get("visibility","public")
    if visibility=="public":return True
    if principal is not None and meta.get("owner")==principal:return True
    if principal is not None and principal in set(meta.get("allowed_users",[])):return True
    allowed_groups=set(meta.get("allowed_groups",[]))
    return bool(allowed_groups and set(groups or [])&allowed_groups)
