from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass

_ARABIC_TO_PERSIAN = str.maketrans({"ي":"ی","ى":"ی","ك":"ک","ۀ":"ه","ة":"ه","ؤ":"و","إ":"ا","أ":"ا"})
_PERSIAN_DIGITS="۰۱۲۳۴۵۶۷۸۹"; _ARABIC_DIGITS="٠١٢٣٤٥٦٧٨٩"; _ASCII_DIGITS="0123456789"
_DIGIT_TRANS=str.maketrans({**dict(zip(_PERSIAN_DIGITS,_ASCII_DIGITS)),**dict(zip(_ARABIC_DIGITS,_ASCII_DIGITS))})
_DIACRITICS_RE=re.compile(r"[\u064b-\u065f\u0670\u06d6-\u06ed]")
_WS_INLINE=set(" \t\f\v\r")


@dataclass(slots=True)
class NormalizedText:
    text: str
    # output char index -> original source char index
    offset_map: list[int]

    def original_span(self,start:int|None,end:int|None)->tuple[int|None,int|None]:
        if start is None or end is None or not self.offset_map:return None,None
        if start >= len(self.offset_map): return None,None
        s=self.offset_map[max(0,start)]
        last_idx=min(max(end-1,start),len(self.offset_map)-1)
        e=self.offset_map[last_idx]+1
        return s,e


@dataclass(slots=True)
class TextNormalizer:
    unicode_form: str = "NFKC"
    normalize_persian: bool = True
    normalize_digits: bool = True
    strip_diacritics: bool = True
    preserve_newlines: bool = True

    @property
    def signature(self)->str:
        raw=json.dumps(asdict(self),sort_keys=True,ensure_ascii=False)
        return hashlib.blake2b(raw.encode(),digest_size=10).hexdigest()

    def _normalize_char(self,ch:str)->str:
        value=unicodedata.normalize(self.unicode_form,ch)
        if self.normalize_persian:value=value.translate(_ARABIC_TO_PERSIAN)
        if self.normalize_digits:value=value.translate(_DIGIT_TRANS)
        if self.strip_diacritics:value=_DIACRITICS_RE.sub("",value)
        return value

    def normalize_with_offsets(self,text:str)->NormalizedText:
        out:list[str]=[]; mapping:list[int]=[]; pending_space:tuple[bool,int]=(False,0); line_has_content=False; newline_run=0

        def emit_space_if_needed():
            nonlocal pending_space
            if pending_space[0] and line_has_content and out and out[-1] not in {" ","\n"}:
                out.append(" "); mapping.append(pending_space[1])
            pending_space=(False,0)

        for idx,ch in enumerate(text):
            value=self._normalize_char(ch)
            for vch in value:
                if not vch:continue
                if vch=="\u200c" and self.normalize_persian:
                    # ZWNJ is meaningful inside Persian words; surrounding spaces are handled separately.
                    emit_space_if_needed(); out.append(vch); mapping.append(idx); line_has_content=True; newline_run=0; continue
                if vch=="\n":
                    pending_space=(False,0)
                    if self.preserve_newlines:
                        # trim a trailing inline space
                        if out and out[-1]==" ": out.pop(); mapping.pop()
                        if line_has_content or (out and out[-1]!="\n"):
                            if newline_run < 2:
                                out.append("\n"); mapping.append(idx); newline_run+=1
                        line_has_content=False
                    else:
                        pending_space=(True,idx)
                    continue
                if vch in _WS_INLINE or vch.isspace():
                    if not pending_space[0]: pending_space=(True,idx)
                    continue
                emit_space_if_needed(); out.append(vch); mapping.append(idx); line_has_content=True; newline_run=0
        # strip trailing whitespace/newlines exactly as normalize() promises
        while out and out[-1] in {" ","\n"}: out.pop(); mapping.pop()
        # strip leading whitespace/newlines
        lead=0
        while lead<len(out) and out[lead] in {" ","\n"}:lead+=1
        if lead:out=out[lead:]; mapping=mapping[lead:]
        return NormalizedText("".join(out),mapping)

    def normalize(self,text:str)->str:
        return self.normalize_with_offsets(text).text

    def normalize_for_search(self,text:str)->str:
        return self.normalize(text).casefold()


DEFAULT_NORMALIZER=TextNormalizer()
