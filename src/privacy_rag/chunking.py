"""Deterministic character-based chunking with stable provenance identifiers."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from privacy_rag.loaders import PageText


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    chunk_id: str
    doc_id: str
    text: str
    source_name: str
    page: int | None
    chunk_index: int
    ingested_at: str
    acl_labels: tuple[str, ...]


def make_doc_id(stable_path: str) -> str:
    """Create a content-free ID from a canonical local path."""
    return "d_" + hashlib.sha256(stable_path.encode("utf-8")).hexdigest()[:24]


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    compact = re.sub(r"\n{3,}", "\n\n", text).strip()
    pieces: list[str] = []
    start = 0
    while start < len(compact):
        end = min(start + chunk_size, len(compact))
        if end < len(compact):
            boundary = max(compact.rfind("\n", start, end), compact.rfind(". ", start, end),
                           compact.rfind(" ", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary + (1 if compact[boundary] == "\n" else 2 if compact[boundary:boundary + 2] == ". " else 0)
        piece = compact[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(compact):
            break
        start = max(start + 1, end - chunk_overlap)
    return pieces


def chunk_pages(pages: list[PageText], *, doc_id: str, source_name: str,
                acl_labels: tuple[str, ...], chunk_size: int, chunk_overlap: int,
                ingested_at: str | None = None) -> list[DocumentChunk]:
    """Split each page separately and assign stable IDs to identical content."""
    timestamp = ingested_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    chunks: list[DocumentChunk] = []
    global_index = 0
    for page in pages:
        for text in _split_text(page.text, chunk_size, chunk_overlap):
            digest = hashlib.sha256(
                f"{doc_id}\0{page.page}\0{global_index}\0{text}".encode("utf-8")
            ).hexdigest()[:24]
            chunks.append(DocumentChunk(
                chunk_id=f"c_{digest}", doc_id=doc_id, text=text, source_name=source_name,
                page=page.page, chunk_index=global_index, ingested_at=timestamp,
                acl_labels=acl_labels,
            ))
            global_index += 1
    return chunks
