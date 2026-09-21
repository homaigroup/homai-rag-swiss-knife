from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Chunk, ChunkingConfig
from ..tokenization import ApproximateTokenCounter, BaseTokenCounter


class BaseChunker(ABC):
    def __init__(self, config: ChunkingConfig, token_counter: BaseTokenCounter | None = None) -> None:
        self.config = config
        self.token_counter = token_counter or ApproximateTokenCounter()

    def count(self, text: str) -> int:
        return self.token_counter.count(text)

    @abstractmethod
    def can_handle(self, text: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def chunk(self, text: str, source_id: str, metadata: dict | None = None) -> list[Chunk]:
        raise NotImplementedError
