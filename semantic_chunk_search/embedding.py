from __future__ import annotations

import hashlib
import math
import re
from collections import OrderedDict
from threading import RLock
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

from .normalization import DEFAULT_NORMALIZER, TextNormalizer

_TOKEN_RE = re.compile(r"[\w\-]+", re.UNICODE)


class BaseEmbeddingProvider(ABC):
    """Embedding contract with separate query/document hooks for asymmetric models."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        return self.embed(text)

    def embed_document(self, text: str) -> list[float]:
        return self.embed(text)

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def embed_query_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_document_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_document(text) for text in texts]

    @property
    @abstractmethod
    def dimensions(self) -> int:
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        return self.__class__.__name__

    @property
    def max_input_tokens(self) -> int | None:
        return None

    @property
    def signature(self) -> str:
        raw = f"{self.model_name}|{self.dimensions}|{self.__class__.__module__}.{self.__class__.__qualname__}"
        return hashlib.blake2b(raw.encode("utf-8"), digest_size=12).hexdigest()


class HashEmbeddingProvider(BaseEmbeddingProvider):
    """
    Deterministic dependency-free fallback.

    This is feature hashing over word/character n-grams. It is intentionally kept
    for tests, offline demos and semantic-chunking fallback. Production retrieval
    should normally use a neural provider such as SentenceTransformerEmbeddingProvider.
    """

    def __init__(self, dimensions: int = 384, normalizer: TextNormalizer | None = None) -> None:
        if dimensions < 32:
            raise ValueError("dimensions must be >= 32")
        self._dimensions = dimensions
        self.normalizer = normalizer or DEFAULT_NORMALIZER

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @staticmethod
    def _index_and_sign(feature: str, dimensions: int) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little", signed=False)
        index = value % dimensions
        sign = 1.0 if ((value >> 63) & 1) == 0 else -1.0
        return index, sign

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        normalized = self.normalizer.normalize_for_search(text)
        if not normalized:
            return vector

        words = _TOKEN_RE.findall(normalized)
        features: list[tuple[str, float]] = []
        for word in words:
            features.append((f"w:{word}", 1.0))
            padded = f"^{word}$"
            for n in (3, 4):
                if len(padded) >= n:
                    for i in range(len(padded) - n + 1):
                        features.append((f"c{n}:{padded[i:i+n]}", 0.35))

        for feature, weight in features:
            idx, sign = self._index_and_sign(feature, self._dimensions)
            vector[idx] += sign * weight

        return l2_normalize(vector)


class SentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):
    """Neural embedding adapter backed by sentence-transformers.

    The default is multilingual E5 so Persian/English mixed corpora work out of
    the box once the optional dependency and model weights are available.
    E5-style query/document prefixes can be disabled for another model.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        *,
        query_prefix: str = "query: ",
        document_prefix: str = "passage: ",
        normalize_embeddings: bool = True,
        device: str | None = None,
        batch_size: int = 32,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "SentenceTransformerEmbeddingProvider requires sentence-transformers. "
                "Install with: pip install 'semantic-chunk-search[neural]'"
            ) from exc
        self._model_name = model_name
        self.revision = revision
        self.local_files_only = local_files_only
        self.model = SentenceTransformer(model_name, device=device, revision=revision, local_files_only=local_files_only)
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        dim = self.model.get_sentence_embedding_dimension()
        if dim is None:
            probe = self.model.encode(["probe"], normalize_embeddings=True)
            dim = len(probe[0])
        self._dimensions = int(dim)
        self._max_input_tokens = int(getattr(self.model, "max_seq_length", 0) or 0) or None

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def max_input_tokens(self) -> int | None:
        return self._max_input_tokens

    @property
    def signature(self) -> str:
        raw = f"{self.model_name}|{self.revision}|{self.dimensions}|{self.query_prefix}|{self.document_prefix}|{self.normalize_embeddings}|{self.max_input_tokens}"
        return hashlib.blake2b(raw.encode("utf-8"), digest_size=12).hexdigest()

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [row.astype(float).tolist() for row in vectors]

    def embed(self, text: str) -> list[float]:
        return self._encode([text])[0]

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.query_prefix + text])[0]

    def embed_document(self, text: str) -> list[float]:
        return self._encode([self.document_prefix + text])[0]

    def embed_query_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode([self.query_prefix + t for t in texts])

    def embed_document_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode([self.document_prefix + t for t in texts])


class CallableEmbeddingProvider(BaseEmbeddingProvider):
    """Adapter for any external embedding API without coupling this package to an SDK."""

    def __init__(
        self,
        dimensions: int,
        document_embed: Callable[[str], list[float]],
        query_embed: Callable[[str], list[float]] | None = None,
        name: str = "external",
    ) -> None:
        self._dimensions = dimensions
        self._document_embed = document_embed
        self._query_embed = query_embed or document_embed
        self._name = name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def signature(self) -> str:
        raw=f"{self._name}|{self._dimensions}"
        return hashlib.blake2b(raw.encode("utf-8"),digest_size=12).hexdigest()

    def embed(self, text: str) -> list[float]:
        return self._document_embed(text)

    def embed_document(self, text: str) -> list[float]:
        return self._document_embed(text)

    def embed_query(self, text: str) -> list[float]:
        return self._query_embed(text)


def l2_normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(v) * float(v) for v in vector))
    if norm == 0.0:
        return [float(v) for v in vector]
    return [float(v) / norm for v in vector]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


class CachingEmbeddingProvider(BaseEmbeddingProvider):
    """LRU cache wrapper keyed by model name, mode and text."""

    def __init__(self, base: BaseEmbeddingProvider, max_entries: int = 10000) -> None:
        self.base = base
        self.max_entries = max_entries
        self._cache: OrderedDict[tuple[str, str, str], list[float]] = OrderedDict(); self._lock=RLock()

    @property
    def dimensions(self) -> int:
        return self.base.dimensions

    @property
    def model_name(self) -> str:
        return self.base.model_name

    @property
    def max_input_tokens(self) -> int | None:
        return self.base.max_input_tokens

    @property
    def signature(self) -> str:
        return self.base.signature

    def _get(self, mode: str, text: str, fn) -> list[float]:
        key = (self.signature, mode, text)
        with self._lock:
            if key in self._cache:
                value = self._cache.pop(key); self._cache[key] = value; return list(value)
        value = fn(text)
        with self._lock:
            self._cache[key] = list(value)
            while len(self._cache) > self.max_entries:self._cache.popitem(last=False)
        return list(value)

    def embed(self, text: str) -> list[float]:
        return self._get("generic", text, self.base.embed)

    def embed_query(self, text: str) -> list[float]:
        return self._get("query", text, self.base.embed_query)

    def embed_document(self, text: str) -> list[float]:
        return self._get("document", text, self.base.embed_document)

    def embed_query_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_document_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_document(t) for t in texts]

    def clear_cache(self) -> None:
        with self._lock:self._cache.clear()
