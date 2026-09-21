from __future__ import annotations

import hashlib
import re

from ..models import Chunk, ChunkType
from ..normalization import DEFAULT_NORMALIZER
from ..tokenization import ApproximateTokenCounter, BaseTokenCounter

_SENTENCE_RE = re.compile(r"(?<=[.!?؟])\s+|\n+(?=\S)")
_DEFAULT_COUNTER = ApproximateTokenCounter()


def estimate_tokens(text: str, counter: BaseTokenCounter | None = None) -> int:
    return (counter or _DEFAULT_COUNTER).count(text)


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_RE.split(text.strip()) if part.strip()]


def stable_chunk_id(source_id: str, source_version: str, chunk_type: ChunkType | str, section_path: list[str], text: str, ordinal: int) -> str:
    raw = "\x1f".join([
        str(source_id), str(source_version), str(getattr(chunk_type, "value", chunk_type)),
        " > ".join(section_path), str(ordinal), DEFAULT_NORMALIZER.normalize_for_search(text),
    ])
    digest = hashlib.blake2b(raw.encode("utf-8"), digest_size=12).hexdigest()
    return f"{source_id}:chunk:{digest}"


def split_by_tokens(text: str, max_tokens: int, counter: BaseTokenCounter | None = None) -> list[str]:
    return (counter or _DEFAULT_COUNTER).split(text, max_tokens)


def _tail_tokens(text: str, overlap_tokens: int, counter: BaseTokenCounter) -> str:
    if overlap_tokens <= 0:
        return ""
    words = text.split()
    # Walk backwards until the requested token budget is reached.
    buf: list[str] = []
    for word in reversed(words):
        candidate = " ".join(reversed([word, *buf])) if buf else word
        if counter.count(candidate) > overlap_tokens and buf:
            break
        buf.insert(0, word)
        if counter.count(" ".join(buf)) >= overlap_tokens:
            break
    return " ".join(buf)


def apply_overlap(texts: list[str], overlap_tokens: int, max_tokens: int, counter: BaseTokenCounter | None = None) -> list[str]:
    if overlap_tokens <= 0 or len(texts) <= 1:
        return texts
    c = counter or _DEFAULT_COUNTER
    out = [texts[0].strip()]
    for text in texts[1:]:
        prefix = _tail_tokens(out[-1], overlap_tokens, c)
        candidate = f"{prefix}\n{text.strip()}".strip() if prefix else text.strip()
        if c.count(candidate) > max_tokens:
            allowed = max(1, max_tokens - c.count(text.strip()))
            prefix = _tail_tokens(out[-1], allowed, c)
            candidate = f"{prefix}\n{text.strip()}".strip() if prefix else text.strip()
            if c.count(candidate) > max_tokens:
                candidate = c.split(candidate, max_tokens)[0]
        out.append(candidate)
    return out


def make_chunks(pieces: list[str], source_id: str, chunk_type: ChunkType, metadata: dict | None = None, counter: BaseTokenCounter | None = None) -> list[Chunk]:
    meta = dict(metadata or {})
    c = counter or _DEFAULT_COUNTER
    source_version = str(meta.get("source_version", "1"))
    section_path = list(meta.get("section_path", []))
    out: list[Chunk] = []
    for i, piece in enumerate(p for p in pieces if p.strip()):
        clean = piece.strip()
        out.append(Chunk(
            raw_text=clean,
            source_id=source_id,
            chunk_index=i,
            chunk_type=chunk_type,
            token_count=c.count(clean),
            metadata=dict(meta),
            id=stable_chunk_id(source_id, source_version, chunk_type, section_path, clean, i),
            parent_id=meta.get("parent_id"),
        ))
    return out


def merge_small_texts(texts: list[str], min_tokens: int, max_tokens: int, counter: BaseTokenCounter | None = None) -> list[str]:
    c = counter or _DEFAULT_COUNTER
    if not texts:
        return []
    out: list[str] = []
    buffer = ""
    for text in texts:
        candidate = f"{buffer}\n\n{text}".strip() if buffer else text.strip()
        if buffer and c.count(candidate) > max_tokens:
            out.append(buffer)
            buffer = text.strip()
        else:
            buffer = candidate
        if c.count(buffer) >= min_tokens:
            out.append(buffer)
            buffer = ""
    if buffer:
        if out and c.count(out[-1] + "\n\n" + buffer) <= max_tokens:
            out[-1] = f"{out[-1]}\n\n{buffer}".strip()
        else:
            out.append(buffer)
    return out
