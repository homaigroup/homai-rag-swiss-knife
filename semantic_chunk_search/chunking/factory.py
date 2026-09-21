from __future__ import annotations

from ..embedding import BaseEmbeddingProvider, HashEmbeddingProvider
from ..models import Chunk, ChunkingConfig
from ..tokenization import ApproximateTokenCounter, BaseTokenCounter, HuggingFaceTokenCounter
from .base import BaseChunker
from .strategies import (
    CodeChunker, CompositeChunker, HierarchicalChunker, MarkdownChunker,
    RecursiveChunker, SemanticChunker, SlidingWindowChunker, TableChunker, TokenBasedChunker,
)


class ChunkingFactory:
    def __init__(self, config: ChunkingConfig | None = None, embedder: BaseEmbeddingProvider | None = None, token_counter: BaseTokenCounter | None = None) -> None:
        self.config = config or ChunkingConfig()
        self.embedder = embedder or HashEmbeddingProvider()
        if token_counter is not None:
            self.token_counter = token_counter
        elif self.config.tokenizer_name:
            self.token_counter = HuggingFaceTokenCounter(self.config.tokenizer_name)
        else:
            self.token_counter = ApproximateTokenCounter()
        c = self.config; t = self.token_counter
        self._chunkers: dict[str, BaseChunker] = {
            "recursive": RecursiveChunker(c, t),
            "semantic": SemanticChunker(c, self.embedder, t),
            "sliding_window": SlidingWindowChunker(c, t),
            "token_based": TokenBasedChunker(c, t),
            "markdown": MarkdownChunker(c, t),
            "hierarchical": HierarchicalChunker(c, t),
            "code": CodeChunker(c, t),
            "table": TableChunker(c, t),
            "composite": CompositeChunker(c, self.embedder, t),
        }

    def register(self, name: str, chunker: BaseChunker) -> None:
        self._chunkers[name] = chunker

    def available_strategies(self) -> list[str]:
        return ["auto", *self._chunkers.keys()]

    def _detect(self, text: str) -> str:
        if self._chunkers["composite"].can_handle(text):
            return "composite"
        if self._chunkers["hierarchical"].can_handle(text):
            return "hierarchical"
        if self._chunkers["table"].can_handle(text):
            return "table"
        if self._chunkers["code"].can_handle(text):
            return "code"
        if self.token_counter.count(text) > self.config.max_chunk_tokens * 5:
            return "sliding_window"
        if self._chunkers["semantic"].can_handle(text):
            return "semantic"
        return "recursive"

    def chunk(self, text: str, source_id: str, strategy: str | None = None, metadata: dict | None = None) -> list[Chunk]:
        if not text or not text.strip():
            raise ValueError("text cannot be empty")
        if not source_id:
            raise ValueError("source_id cannot be empty")
        selected = strategy or self.config.strategy
        if selected == "auto": selected = self._detect(text)
        if selected not in self._chunkers:
            raise ValueError(f"unknown strategy: {selected}")
        merged_meta = dict(metadata or {}); merged_meta.setdefault("strategy", selected)
        chunks = self._chunkers[selected].chunk(text, source_id, merged_meta)
        cursor = 0
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
            # Best-effort provenance offsets in the original input.
            needle = chunk.raw_text.strip()
            pos = text.find(needle, cursor) if needle else -1
            if pos < 0 and needle:
                pos = text.find(needle)
            if pos >= 0:
                chunk.start_char = pos; chunk.end_char = pos + len(needle); cursor = chunk.end_char
        return chunks
