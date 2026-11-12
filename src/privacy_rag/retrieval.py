"""Local Chroma storage and ACL-filtered vector retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from privacy_rag.chunking import DocumentChunk
from privacy_rag.config import Settings
from privacy_rag.privacy import acl_metadata_key

_COLLECTION_NAME = "private_knowledge"
_STORE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    doc_id: str
    source_name: str
    page: int | None
    acl_labels: tuple[str, ...]


class LocalVectorStore:
    """Persistent local Chroma collection; all queries filter ACLs before ranking."""

    def __init__(self, settings: Settings, embedding_dimension: int | None) -> None:
        path: Path = settings.vector_store_path
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(path),
            settings=ChromaSettings(anonymized_telemetry=False, is_persistent=True),
        )
        expected = {
            "hnsw:space": "cosine",
            "embed_model": settings.ollama_embed_model,
            "schema_version": _STORE_SCHEMA_VERSION,
        }
        if embedding_dimension is None:
            try:
                self._collection = self._client.get_collection(
                    name=_COLLECTION_NAME, embedding_function=None,
                )
            except ValueError:
                self._collection = None
                return
        else:
            self._collection = self._client.get_or_create_collection(
                name=_COLLECTION_NAME,
                metadata={**expected, "embed_dim": embedding_dimension},
                embedding_function=None,
            )
        actual = self._collection.metadata or {}
        for key, value in expected.items():
            if actual.get(key) != value:
                raise RuntimeError("Vector index metadata does not match configured embedding model")
        actual_dimension = actual.get("embed_dim")
        if not isinstance(actual_dimension, int) or actual_dimension < 1:
            raise RuntimeError("Vector index has invalid embedding dimension metadata")
        if embedding_dimension is not None and actual_dimension != embedding_dimension:
            raise RuntimeError("Vector index metadata does not match configured embedding model")

    @staticmethod
    def _acl_filter(allowed_groups: tuple[str, ...]) -> dict[str, Any]:
        if not allowed_groups or len(allowed_groups) > 32:
            raise ValueError("Request has no valid ACL groups")
        filters = [{acl_metadata_key(group): True} for group in sorted(set(allowed_groups))]
        return filters[0] if len(filters) == 1 else {"$or": filters}

    def query(self, vector: list[float], *, allowed_groups: tuple[str, ...],
              top_k: int) -> list[RetrievedChunk]:
        """Search Chroma with the authorization filter in the vector query itself."""
        if self._collection is None:
            return []
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=self._acl_filter(allowed_groups),
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        found: list[RetrievedChunk] = []
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            if not isinstance(metadata, dict) or not isinstance(text, str):
                continue
            try:
                labels_raw = json.loads(str(metadata.get("acl_labels", "[]")))
                labels = tuple(str(label) for label in labels_raw if isinstance(label, str))
                page_value = int(metadata.get("page", -1))
                score = 1.0 - float(distance)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            found.append(RetrievedChunk(
                chunk_id=str(chunk_id), text=text, score=score,
                doc_id=str(metadata.get("doc_id", "")),
                source_name=str(metadata.get("source_name", "document")),
                page=page_value if page_value > 0 else None,
                acl_labels=labels,
            ))
        return found

    def upsert_document(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> int:
        """Insert new chunks, then remove stale versions for this document."""
        if self._collection is None:
            raise RuntimeError("Vector collection is not initialized")
        if len(chunks) != len(vectors):
            raise ValueError("Chunk and vector counts do not match")
        if not chunks:
            return 0
        doc_id = chunks[0].doc_id
        if any(chunk.doc_id != doc_id for chunk in chunks):
            raise ValueError("A document update must contain one document ID")
        old = self._collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        old_ids = set(old.get("ids", []))
        new_ids = {chunk.chunk_id for chunk in chunks}
        batch_size = 256
        for start in range(0, len(chunks), batch_size):
            chunk_batch = chunks[start:start + batch_size]
            vector_batch = vectors[start:start + batch_size]
            self._collection.upsert(
                ids=[chunk.chunk_id for chunk in chunk_batch],
                embeddings=vector_batch,
                documents=[chunk.text for chunk in chunk_batch],
                metadatas=[self._metadata(chunk, len(vector_batch[0])) for chunk in chunk_batch],
            )
        stale = old_ids - new_ids
        if stale:
            self._collection.delete(ids=sorted(stale))
        return len(chunks)

    def delete_document(self, doc_id: str, *, allowed_groups: tuple[str, ...]) -> int:
        if self._collection is None:
            return 0
        current = self._collection.get(
            where={"$and": [{"doc_id": doc_id}, self._acl_filter(allowed_groups)]},
            include=["metadatas"],
        )
        ids = current.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def documents(self, *, allowed_groups: tuple[str, ...]) -> list[dict[str, Any]]:
        if self._collection is None:
            return []
        current = self._collection.get(
            where=self._acl_filter(allowed_groups), include=["metadatas"],
        )
        entries: dict[str, dict[str, Any]] = {}
        for metadata in current.get("metadatas", []) or []:
            if not isinstance(metadata, dict):
                continue
            doc_id = str(metadata.get("doc_id", ""))
            if not doc_id:
                continue
            if doc_id not in entries:
                entries[doc_id] = {
                    "doc_id": doc_id,
                    "source_name": str(metadata.get("source_name", "document")),
                    "chunks": 0,
                    "acl_labels": json.loads(str(metadata.get("acl_labels", "[]"))),
                }
            entries[doc_id]["chunks"] += 1
        return sorted(entries.values(), key=lambda item: item["doc_id"])

    @staticmethod
    def _metadata(chunk: DocumentChunk, dimension: int) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "doc_id": chunk.doc_id,
            "source_name": chunk.source_name,
            "page": chunk.page if chunk.page is not None else -1,
            "chunk_index": chunk.chunk_index,
            "ingested_at": chunk.ingested_at,
            "acl_labels": json.dumps(list(chunk.acl_labels), separators=(",", ":")),
            "embed_dim": dimension,
        }
        for label in chunk.acl_labels:
            metadata[acl_metadata_key(label)] = True
        return metadata
