from __future__ import annotations

import ast
import hashlib
import math
import re

from .base import BaseChunker
from .utils import apply_overlap, make_chunks, merge_small_texts, split_by_tokens, split_sentences, stable_chunk_id
from ..embedding import BaseEmbeddingProvider, cosine_similarity
from ..models import Chunk, ChunkType


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct / 100.0
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - idx) + ordered[hi] * (idx - lo)


class TokenBasedChunker(BaseChunker):
    def can_handle(self, text: str) -> bool:
        return bool(text.strip())

    def chunk(self, text: str, source_id: str, metadata: dict | None = None) -> list[Chunk]:
        words = text.split()
        if not words:
            return []
        step_words = max(1, self.config.max_chunk_tokens - self.config.overlap_tokens)
        # Use token counter to enforce final size even though stepping is approximate.
        pieces: list[str] = []
        start = 0
        while start < len(words):
            piece_words: list[str] = []
            i = start
            while i < len(words):
                candidate = " ".join([*piece_words, words[i]])
                if piece_words and self.count(candidate) > self.config.max_chunk_tokens:
                    break
                piece_words.append(words[i]); i += 1
            if not piece_words:
                piece_words = [words[start]]; i = start + 1
            pieces.append(" ".join(piece_words))
            if i >= len(words):
                break
            overlap_text = " ".join(piece_words)
            overlap_words = overlap_text.split()[-self.config.overlap_tokens:] if self.config.overlap_tokens else []
            start = max(start + 1, i - len(overlap_words))
        return make_chunks(pieces, source_id, ChunkType.TOKEN_BASED, metadata, self.token_counter)


class SlidingWindowChunker(TokenBasedChunker):
    def chunk(self, text: str, source_id: str, metadata: dict | None = None) -> list[Chunk]:
        chunks = super().chunk(text, source_id, metadata)
        for chunk in chunks:
            chunk.chunk_type = ChunkType.SLIDING_WINDOW
        return chunks


class RecursiveChunker(BaseChunker):
    separators = ("\n\n", "\n", ". ", " ")

    def can_handle(self, text: str) -> bool:
        return bool(text.strip())

    def _split(self, text: str, depth: int = 0) -> list[str]:
        if self.count(text) <= self.config.max_chunk_tokens:
            return [text.strip()]
        if depth >= len(self.separators):
            return split_by_tokens(text, self.config.max_chunk_tokens, self.token_counter)
        sep = self.separators[depth]
        parts = [p.strip() for p in text.split(sep) if p.strip()]
        if len(parts) <= 1:
            return self._split(text, depth + 1)
        result: list[str] = []
        buffer = ""
        for part in parts:
            candidate = f"{buffer}{sep}{part}".strip() if buffer else part
            if self.count(candidate) <= self.config.max_chunk_tokens:
                buffer = candidate
            else:
                if buffer:
                    result.extend(self._split(buffer, depth + 1))
                buffer = part
        if buffer:
            result.extend(self._split(buffer, depth + 1))
        return result

    def chunk(self, text: str, source_id: str, metadata: dict | None = None) -> list[Chunk]:
        pieces = merge_small_texts(self._split(text.strip()), self.config.min_chunk_tokens, self.config.max_chunk_tokens, self.token_counter)
        pieces = apply_overlap(pieces, self.config.overlap_tokens, self.config.max_chunk_tokens, self.token_counter)
        return make_chunks(pieces, source_id, ChunkType.RECURSIVE, metadata, self.token_counter)


class SemanticChunker(BaseChunker):
    def __init__(self, config, embedder: BaseEmbeddingProvider, token_counter=None) -> None:
        super().__init__(config, token_counter)
        self.embedder = embedder

    def can_handle(self, text: str) -> bool:
        return len(split_sentences(text)) >= 2

    def chunk(self, text: str, source_id: str, metadata: dict | None = None) -> list[Chunk]:
        sentences = split_sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            pieces = split_by_tokens(sentences[0], self.config.max_chunk_tokens, self.token_counter)
            return make_chunks(pieces, source_id, ChunkType.SEMANTIC, metadata, self.token_counter)
        vectors = self.embedder.embed_batch(sentences)
        similarities = [cosine_similarity(vectors[i - 1], vectors[i]) for i in range(1, len(vectors))]
        threshold = self.config.semantic_threshold
        if self.config.adaptive_semantic and similarities:
            threshold = _percentile(similarities, self.config.semantic_break_percentile)
        groups: list[str] = []
        current = sentences[0]
        for i in range(1, len(sentences)):
            similarity = similarities[i - 1]
            candidate = f"{current} {sentences[i]}".strip()
            too_large = self.count(candidate) > self.config.max_chunk_tokens
            topic_shift = similarity <= threshold
            if too_large or (topic_shift and self.count(current) >= self.config.min_chunk_tokens):
                groups.append(current); current = sentences[i]
            else:
                current = candidate
        if current:
            groups.append(current)
        groups = merge_small_texts(groups, self.config.min_chunk_tokens, self.config.max_chunk_tokens, self.token_counter)
        final: list[str] = []
        for group in groups:
            final.extend(split_by_tokens(group, self.config.max_chunk_tokens, self.token_counter) if self.count(group) > self.config.max_chunk_tokens else [group])
        final = apply_overlap(final, self.config.overlap_tokens, self.config.max_chunk_tokens, self.token_counter)
        meta = dict(metadata or {}); meta["semantic_threshold_used"] = threshold
        return make_chunks(final, source_id, ChunkType.SEMANTIC, meta, self.token_counter)


class MarkdownChunker(RecursiveChunker):
    _heading = re.compile(r"(?m)^#{1,6}\s+.+$")
    def can_handle(self, text: str) -> bool:
        return bool(self._heading.search(text))
    def chunk(self, text: str, source_id: str, metadata: dict | None = None) -> list[Chunk]:
        chunks = super().chunk(text, source_id, metadata)
        for c in chunks: c.chunk_type = ChunkType.MARKDOWN
        return chunks


class TableChunker(BaseChunker):
    _row = re.compile(r"^\s*\|.*\|\s*$")
    def can_handle(self, text: str) -> bool:
        return sum(1 for line in text.splitlines() if self._row.match(line)) >= 2
    def chunk(self, text: str, source_id: str, metadata: dict | None = None) -> list[Chunk]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines: return []
        header = lines[:2] if len(lines) >= 2 and re.search(r"-{3,}", lines[1]) else lines[:1]
        rows = lines[len(header):]
        pieces: list[str] = []
        buf = "\n".join(header)
        for row in rows:
            cand = f"{buf}\n{row}"
            if self.count(cand) > self.config.max_chunk_tokens and buf != "\n".join(header):
                pieces.append(buf); buf = "\n".join([*header, row])
            else: buf = cand
        if buf: pieces.append(buf)
        meta = dict(metadata or {}); meta["block_type"] = "table"
        return make_chunks(pieces, source_id, ChunkType.TABLE, meta, self.token_counter)


class CodeChunker(BaseChunker):
    _fenced = re.compile(r"```[\s\S]*?```", re.MULTILINE)
    _code_hint = re.compile(r"(?m)^\s*(def |class |async def |function |import |from |const |let |var |public |private |#include )")
    def can_handle(self, text: str) -> bool:
        return bool(self._fenced.search(text) or self._code_hint.search(text))

    def _python_ast_units(self, text: str) -> list[str]:
        source = re.sub(r"^```(?:python|py)?\s*|\s*```$", "", text.strip(), flags=re.I)
        try: tree = ast.parse(source)
        except SyntaxError: return []
        lines = source.splitlines()
        units: list[str] = []
        prelude_end = min((n.lineno for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))), default=len(lines)+1)
        prelude = "\n".join(lines[:prelude_end-1]).strip()
        if prelude: units.append(prelude)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and hasattr(node, "end_lineno"):
                units.append("\n".join(lines[node.lineno-1:node.end_lineno]).strip())
        return [u for u in units if u]

    def chunk(self, text: str, source_id: str, metadata: dict | None = None) -> list[Chunk]:
        units = self._python_ast_units(text)
        if not units:
            units = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
        pieces: list[str] = []
        for unit in units:
            pieces.extend(split_by_tokens(unit, self.config.max_chunk_tokens, self.token_counter) if self.count(unit) > self.config.max_chunk_tokens else [unit])
        pieces = apply_overlap(pieces, min(self.config.overlap_tokens, max(0, self.config.max_chunk_tokens//6)), self.config.max_chunk_tokens, self.token_counter)
        meta = dict(metadata or {}); meta["block_type"] = "code"; meta["ast_aware"] = bool(self._python_ast_units(text))
        return make_chunks(pieces, source_id, ChunkType.CODE, meta, self.token_counter)


class HierarchicalChunker(BaseChunker):
    _heading_line = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
    def can_handle(self, text: str) -> bool:
        return bool(re.search(r"(?m)^#{1,6}\s+.+$", text))
    @staticmethod
    def _parent_id(source_id: str, source_version: str, path: list[str]) -> str:
        raw = f"{source_id}::{source_version}::{' > '.join(path)}"
        return f"{source_id}:section:{hashlib.blake2b(raw.encode(), digest_size=10).hexdigest()}"

    def chunk(self, text: str, source_id: str, metadata: dict | None = None) -> list[Chunk]:
        base_meta = dict(metadata or {}); version = str(base_meta.get("source_version", "1"))
        heading_stack: list[tuple[int, str]] = []; sections: list[tuple[list[str], list[str]]] = []; body: list[str] = []; current_path: list[str] = []
        def flush():
            nonlocal body
            if "\n".join(body).strip(): sections.append((list(current_path), list(body)))
            body = []
        for line in text.splitlines():
            match = self._heading_line.match(line)
            if match:
                flush(); level=len(match.group(1)); title=match.group(2).strip()
                while heading_stack and heading_stack[-1][0] >= level: heading_stack.pop()
                heading_stack.append((level,title)); current_path=[name for _,name in heading_stack]
            else: body.append(line)
        flush()
        recursive = RecursiveChunker(self.config, self.token_counter); chunks: list[Chunk] = []
        first_h1 = re.search(r"(?m)^#\s+(.+)$", text); document_title = first_h1.group(1).strip() if first_h1 else ""
        for path, lines in sections:
            body_text="\n".join(lines).strip(); leaf=path[-1] if path else ""; section_text=f"{leaf}\n\n{body_text}".strip() if leaf else body_text
            if not section_text: continue
            pieces=merge_small_texts(recursive._split(section_text), self.config.min_chunk_tokens, self.config.max_chunk_tokens, self.token_counter)
            pieces=apply_overlap(pieces, self.config.overlap_tokens, self.config.max_chunk_tokens, self.token_counter)
            parent_id=self._parent_id(source_id, version, path or ["root"]); parent_context=" > ".join(path)
            for local_i, child in enumerate(pieces):
                meta=dict(base_meta); meta.update({"section_path":list(path),"parent_id":parent_id,"parent_context":parent_context,"parent_text":section_text})
                if document_title: meta.setdefault("document_title",document_title)
                chunks.append(Chunk(child.strip(),source_id,len(chunks),ChunkType.HIERARCHICAL,self.count(child),meta,
                    stable_chunk_id(source_id,version,ChunkType.HIERARCHICAL,path,child,local_i),parent_id=parent_id))
        return chunks


class CompositeChunker(BaseChunker):
    """Parse mixed Markdown/prose/code/table documents and route each block independently."""
    _heading = re.compile(r"^(#{1,6})\s+(.+)$")
    _fence_start = re.compile(r"^```(\w+)?\s*$")
    _table = re.compile(r"^\s*\|.*\|\s*$")
    def __init__(self, config, embedder: BaseEmbeddingProvider, token_counter=None) -> None:
        super().__init__(config, token_counter); self.embedder=embedder
    def can_handle(self,text:str)->bool:
        kinds=sum(bool(x) for x in (re.search(r"(?m)^#{1,6}\s",text), re.search(r"```",text), re.search(r"(?m)^\s*\|.*\|\s*$",text)))
        return kinds>=2
    def chunk(self,text:str,source_id:str,metadata:dict|None=None)->list[Chunk]:
        lines=text.splitlines(); blocks:list[tuple[str,str,list[str]]]=[]; path:list[str]=[]; stack:list[tuple[int,str]]=[]; prose:list[str]=[]; i=0
        def flush_prose():
            nonlocal prose
            content="\n".join(prose).strip()
            if content: blocks.append(("prose",content,list(path)))
            prose=[]
        while i<len(lines):
            line=lines[i]; hm=self._heading.match(line)
            if hm:
                flush_prose(); level=len(hm.group(1)); title=hm.group(2).strip()
                while stack and stack[-1][0]>=level: stack.pop()
                stack.append((level,title)); path=[x for _,x in stack]; i+=1; continue
            fm=self._fence_start.match(line)
            if fm:
                flush_prose(); code=[line]; i+=1
                while i<len(lines):
                    code.append(lines[i]);
                    if lines[i].strip().startswith("```"): i+=1; break
                    i+=1
                blocks.append(("code","\n".join(code),list(path))); continue
            if self._table.match(line):
                flush_prose(); rows=[]
                while i<len(lines) and self._table.match(lines[i]): rows.append(lines[i]); i+=1
                blocks.append(("table","\n".join(rows),list(path))); continue
            prose.append(line); i+=1
        flush_prose(); out:list[Chunk]=[]; version=str((metadata or {}).get("source_version","1"))
        # Parent text is the complete structural section, not whichever child happened to be indexed first.
        section_texts:dict[tuple[str,...],list[str]]={}
        for _kind,_content,_path in blocks:
            section_texts.setdefault(tuple(_path or ["root"]),[]).append(_content)
        for block_type,content,bpath in blocks:
            meta=dict(metadata or {}); meta.update({"section_path":bpath,"block_type":block_type})
            parent_id=HierarchicalChunker._parent_id(source_id,version,bpath or ["root"]); meta["parent_id"]=parent_id; meta["parent_context"]=" > ".join(bpath); meta["parent_text"]="\n\n".join(section_texts.get(tuple(bpath or ["root"]),[content])).strip()
            if block_type=="code": child=CodeChunker(self.config,self.token_counter).chunk(content,source_id,meta)
            elif block_type=="table": child=TableChunker(self.config,self.token_counter).chunk(content,source_id,meta)
            else:
                semantic=SemanticChunker(self.config,self.embedder,self.token_counter)
                child=semantic.chunk(content,source_id,meta) if semantic.can_handle(content) else RecursiveChunker(self.config,self.token_counter).chunk(content,source_id,meta)
            for c in child:
                c.chunk_type=ChunkType.COMPOSITE; c.chunk_index=len(out); c.parent_id=parent_id
                c.id=stable_chunk_id(source_id,version,ChunkType.COMPOSITE,bpath,c.raw_text,len(out)); out.append(c)
        return out


class TreeSitterCodeChunker(BaseChunker):
    """Generic multi-language AST chunker using an injected tree-sitter-compatible parser.

    The parser must expose ``parse(bytes)`` and nodes with start_byte/end_byte/type/children.
    This avoids hard-coupling the core package to language grammar wheels.
    """
    def __init__(self, config, parser, node_types: set[str] | None = None, token_counter=None) -> None:
        super().__init__(config, token_counter)
        self.parser=parser
        self.node_types=node_types or {"function_definition","class_definition","method_definition","function_declaration","class_declaration","interface_declaration"}

    def can_handle(self,text:str)->bool:return bool(text.strip())

    def chunk(self,text:str,source_id:str,metadata:dict|None=None)->list[Chunk]:
        raw=text.encode("utf-8"); tree=self.parser.parse(raw); units=[]
        def walk(node):
            if getattr(node,"type","") in self.node_types:
                units.append(raw[node.start_byte:node.end_byte].decode("utf-8",errors="replace"))
                return
            for child in getattr(node,"children",[]): walk(child)
        walk(tree.root_node)
        if not units: units=[text]
        pieces=[]
        for unit in units:
            pieces.extend(split_by_tokens(unit,self.config.max_chunk_tokens,self.token_counter) if self.count(unit)>self.config.max_chunk_tokens else [unit])
        meta=dict(metadata or {}); meta["ast_aware"]=True; meta["parser"]="tree_sitter"
        return make_chunks(apply_overlap(pieces,self.config.overlap_tokens,self.config.max_chunk_tokens,self.token_counter),source_id,ChunkType.CODE,meta,self.token_counter)
