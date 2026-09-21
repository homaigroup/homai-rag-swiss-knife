from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
import re

from .normalization import DEFAULT_NORMALIZER, TextNormalizer

_TOKEN_RE=re.compile(r"[\w\-]+|[^\w\s]",re.UNICODE)

class BaseTokenCounter(ABC):
    @abstractmethod
    def count(self,text:str)->int:...
    @abstractmethod
    def split(self,text:str,max_tokens:int)->list[str]:...
    @property
    def signature(self)->str:return self.__class__.__name__

@dataclass(slots=True)
class ApproximateTokenCounter(BaseTokenCounter):
    normalizer:TextNormalizer=field(default_factory=lambda:DEFAULT_NORMALIZER)
    def tokens(self,text:str)->list[str]:return _TOKEN_RE.findall(self.normalizer.normalize(text))
    def count(self,text:str)->int:return len(self.tokens(text))
    def split(self,text:str,max_tokens:int)->list[str]:
        if max_tokens<1:raise ValueError("max_tokens must be >= 1")
        words=text.split()
        if not words:return []
        out=[]; buf=[]
        for word in words:
            candidate=" ".join([*buf,word])
            if buf and self.count(candidate)>max_tokens:out.append(" ".join(buf));buf=[word]
            else:buf.append(word)
        if buf:out.append(" ".join(buf))
        return out
    @property
    def signature(self)->str:return f"approx:{self.normalizer.signature}"

class TokenizerObjectCounter(BaseTokenCounter):
    """Counter backed by any HF-like tokenizer exposing encode/decode."""
    def __init__(self,tokenizer,name:str="tokenizer") -> None:self.tokenizer=tokenizer;self.name=name
    def count(self,text:str)->int:return len(self.tokenizer.encode(text,add_special_tokens=False))
    def split(self,text:str,max_tokens:int)->list[str]:
        if max_tokens<1:raise ValueError("max_tokens must be >= 1")
        ids=self.tokenizer.encode(text,add_special_tokens=False)
        return [self.tokenizer.decode(ids[i:i+max_tokens],skip_special_tokens=True) for i in range(0,len(ids),max_tokens)]
    @property
    def signature(self)->str:return f"tokenizer:{self.name}"

class HuggingFaceTokenCounter(TokenizerObjectCounter):
    def __init__(self,tokenizer_name:str)->None:
        try:from transformers import AutoTokenizer
        except ImportError as exc:raise ImportError("Install transformers to use HuggingFaceTokenCounter") from exc
        super().__init__(AutoTokenizer.from_pretrained(tokenizer_name),tokenizer_name)

class TiktokenTokenCounter(BaseTokenCounter):
    def __init__(self,encoding_name:str="cl100k_base")->None:
        try:import tiktoken
        except ImportError as exc:raise ImportError("Install tiktoken to use TiktokenTokenCounter") from exc
        self.encoding=tiktoken.get_encoding(encoding_name);self.encoding_name=encoding_name
    def count(self,text:str)->int:return len(self.encoding.encode(text))
    def split(self,text:str,max_tokens:int)->list[str]:
        if max_tokens<1:raise ValueError("max_tokens must be >= 1")
        ids=self.encoding.encode(text);return [self.encoding.decode(ids[i:i+max_tokens]) for i in range(0,len(ids),max_tokens)]
    @property
    def signature(self)->str:return f"tiktoken:{self.encoding_name}"

def truncate_tokens(text:str,budget:int,counter:BaseTokenCounter)->str:
    if budget<=0:return ""
    if counter.count(text)<=budget:return text
    pieces=counter.split(text,budget);return pieces[0] if pieces else ""
