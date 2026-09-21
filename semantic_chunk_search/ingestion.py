from __future__ import annotations

from dataclasses import dataclass,field
from html.parser import HTMLParser
import re
from pathlib import Path
from typing import Protocol

@dataclass(slots=True)
class DocumentBlock:
    text:str
    page:int|None=None
    block_type:str="text"
    bbox:tuple[float,float,float,float]|None=None
    metadata:dict=field(default_factory=dict)

@dataclass(slots=True)
class LoadedDocument:
    pages:list[str]
    blocks:list[DocumentBlock]=field(default_factory=list)
    metadata:dict=field(default_factory=dict)

class DocumentLoader(Protocol):
    def load(self,path:str|Path)->LoadedDocument:...

class OCRProvider(Protocol):
    def extract_text(self,source)->str:...

class PlainTextLoader:
    def load(self,path:str|Path)->LoadedDocument:
        p=Path(path);text=p.read_text(encoding="utf-8");return LoadedDocument([text],[DocumentBlock(text,page=1)],{"filename":p.name,"format":"text"})

class _HTMLTextParser(HTMLParser):
    BLOCKS={"p","div","section","article","li","h1","h2","h3","h4","h5","h6","pre","code","td","th"}
    _HIDDEN_STYLE=re.compile(r"(?:display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0(?:px|em|rem|%)?|opacity\s*:\s*0(?:\D|$))",re.I)
    def __init__(self):
        super().__init__();self.buf=[];self.blocks=[];self._hidden=0;self._stack=[]
    @classmethod
    def _is_suppressed(cls,tag,attrs):
        attrs={str(k).casefold():("" if v is None else str(v)) for k,v in attrs}
        if tag in {"script","style","noscript","template"}:return True
        if "hidden" in attrs:return True
        if attrs.get("aria-hidden","").strip().casefold()=="true":return True
        return bool(cls._HIDDEN_STYLE.search(attrs.get("style","")))
    def handle_starttag(self,tag,attrs):
        tag=tag.casefold()
        if tag in self.BLOCKS and self.buf and self._hidden==0:self._flush()
        suppressed=self._is_suppressed(tag,attrs);self._stack.append((tag,suppressed))
        if suppressed:self._hidden+=1
    def handle_startendtag(self,tag,attrs):
        tag=tag.casefold()
        if tag in self.BLOCKS and self.buf and self._hidden==0:self._flush()
        # Self-closing hidden content has no data payload to retain.
    def handle_endtag(self,tag):
        tag=tag.casefold()
        if tag in self.BLOCKS and self._hidden==0:self._flush()
        # Pop through the matching tag so malformed nested HTML cannot leave us
        # permanently in a hidden state.
        while self._stack:
            opened,suppressed=self._stack.pop()
            if suppressed and self._hidden:self._hidden-=1
            if opened==tag:break
    def handle_data(self,data):
        if not self._hidden:self.buf.append(data)
    def _flush(self):
        text=" ".join(" ".join(self.buf).split()).strip();self.buf=[]
        if text:self.blocks.append(text)
    def close(self):super().close();self._flush()

class HTMLLoader:
    def load(self,path:str|Path)->LoadedDocument:
        p=Path(path);parser=_HTMLTextParser();parser.feed(p.read_text(encoding="utf-8",errors="replace"));parser.close();text="\n\n".join(parser.blocks)
        return LoadedDocument([text],[DocumentBlock(x,page=1,block_type="html_block") for x in parser.blocks],{"filename":p.name,"format":"html"})

class PDFLoader:
    """Optional text PDF loader. For layout/OCR use a custom DocumentLoader that supplies blocks/bboxes."""
    def __init__(self,ocr:OCRProvider|None=None)->None:self.ocr=ocr
    def load(self,path:str|Path)->LoadedDocument:
        try:from pypdf import PdfReader
        except ImportError as exc:raise ImportError("Install pypdf or semantic-chunk-search[documents]") from exc
        p=Path(path);reader=PdfReader(str(p));pages=[];blocks=[]
        for i,page in enumerate(reader.pages,1):
            text=(page.extract_text() or "").strip()
            if not text and self.ocr is not None:
                try:
                    if hasattr(self.ocr,"extract_pdf_page"):text=str(self.ocr.extract_pdf_page(str(p),i) or "").strip()
                    else:text=str(self.ocr.extract_text(page) or "").strip()
                except Exception:text=""
            pages.append(text)
            if text:blocks.append(DocumentBlock(text,page=i,block_type="page",metadata={"ocr":bool(self.ocr and not (page.extract_text() or "").strip())}))
        return LoadedDocument(pages,blocks,{"filename":p.name,"format":"pdf"})

class DOCXLoader:
    def load(self,path:str|Path)->LoadedDocument:
        try:from docx import Document
        except ImportError as exc:raise ImportError("Install python-docx or semantic-chunk-search[documents]") from exc
        p=Path(path);doc=Document(str(p));parts=[];blocks=[]
        for para in doc.paragraphs:
            text=para.text.strip()
            if text:
                style=getattr(para.style,"name","");kind="heading" if str(style).casefold().startswith("heading") else "paragraph"
                parts.append(text);blocks.append(DocumentBlock(text,page=1,block_type=kind,metadata={"style":style}))
        for table_no,table in enumerate(doc.tables,1):
            rows=[]
            for row in table.rows:
                cells=[" ".join(cell.text.split()) for cell in row.cells];rows.append(" | ".join(cells))
            text="\n".join(x for x in rows if x.strip())
            if text:parts.append(text);blocks.append(DocumentBlock(text,page=1,block_type="table",metadata={"table_index":table_no}))
        text="\n\n".join(parts);return LoadedDocument([text],blocks,{"filename":p.name,"format":"docx"})

def loader_for_path(path:str|Path)->DocumentLoader:
    suffix=Path(path).suffix.casefold()
    if suffix in {".txt",".md",".json",".csv"}:return PlainTextLoader()
    if suffix in {".html",".htm"}:return HTMLLoader()
    if suffix==".pdf":return PDFLoader()
    if suffix==".docx":return DOCXLoader()
    raise ValueError(f"unsupported document type: {suffix or '<none>'}")
