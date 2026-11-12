"""Application services for ingestion, ACL-filtered retrieval, and answers."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from privacy_rag.chunking import chunk_pages, make_doc_id
from privacy_rag.config import Settings
from privacy_rag.embeddings import OllamaClient
from privacy_rag.graph import build_graph
from privacy_rag.loaders import SUPPORTED_EXTENSIONS, load_document
from privacy_rag.privacy import principal_groups, sanitize_source_name
from privacy_rag.retrieval import LocalVectorStore

_DOC_ID_RE = re.compile(r"^d_[a-f0-9]{24}$")
_GROUP_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,63}$")
logger = logging.getLogger(__name__)


class RAGApplication:
    """Own local clients and expose the application's RAG operations."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ollama: OllamaClient | None = None
        self.store: LocalVectorStore | None = None
        self.graph: Any | None = None

    def _ensure_rag(self) -> None:
        if self.graph is not None:
            return
        client = OllamaClient(self.settings)
        try:
            model_names = [self.settings.ollama_embed_model, self.settings.ollama_chat_model]
            if self.settings.ollama_verify_model:
                model_names.append(self.settings.ollama_verify_model)
            client.require_installed_local_models(tuple(model_names))
            probe = client.embed(["local embedding dimension probe"], query=True)
            if not probe:
                raise RuntimeError("Embedding model returned no vector")
            store = LocalVectorStore(self.settings, len(probe[0]))
            graph = build_graph(self.settings, client, store)
        except BaseException:
            client.close()
            raise
        self.ollama = client
        self.store = store
        self.graph = graph

    def _ensure_store(self) -> LocalVectorStore:
        if self.store is None:
            self.store = LocalVectorStore(self.settings, None)
        return self.store

    def close(self) -> None:
        if self.ollama is not None:
            self.ollama.close()

    def model_metadata(self) -> dict[str, dict[str, str]]:
        """Return configured local model details for measured benchmark records."""
        self._ensure_rag()
        if self.ollama is None:
            raise RuntimeError("Local model client was not initialized")
        chat_model = self.settings.ollama_chat_model
        verifier_model = self.settings.ollama_verify_model or chat_model
        return {
            "chat": {"name": chat_model, **self.ollama.model_details(chat_model)},
            "verifier": {"name": verifier_model, **self.ollama.model_details(verifier_model)},
            "embedding": {"name": self.settings.ollama_embed_model},
        }

    def __enter__(self) -> "RAGApplication":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def current_principal() -> str:
        """Read the process identity from the OS, not user-overridable name variables."""
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            account_name = ctypes.create_unicode_buffer(256)
            account_length = wintypes.DWORD(len(account_name))
            advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
            get_user_name = advapi32.GetUserNameW
            get_user_name.argtypes = [wintypes.LPWSTR, wintypes.LPDWORD]
            get_user_name.restype = wintypes.BOOL
            if not get_user_name(account_name, ctypes.byref(account_length)):
                raise OSError(ctypes.get_last_error(), "Could not read the current OS account")
            return account_name.value
        import pwd

        return pwd.getpwuid(os.geteuid()).pw_name

    def current_access_groups(self) -> tuple[str, ...]:
        """Resolve groups from the OS account running this local application."""
        return principal_groups(self.current_principal(), self.settings.acl_principals)

    def ask(self, question: str) -> dict[str, object]:
        if not isinstance(question, str) or not question.strip() or len(question.strip()) > 4000:
            return {
                "status": "rejected",
                "answer": "The request is invalid or this local account is not authorized to ask it.",
                "sources": [],
            }
        self._ensure_rag()
        user = self.current_principal()
        groups = principal_groups(user, self.settings.acl_principals)
        started = time.perf_counter()
        assert self.graph is not None
        state = self.graph.invoke({"question": question, "allowed_groups": groups})
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        metrics = state.get("metrics", {})
        final = state.get("final") or {
            "status": "abstained",
            "answer": "The approved knowledge base does not contain enough verified evidence to answer.",
            "sources": [],
            "reason": "claims_not_verified",
        }
        final = dict(final)
        final["diagnostics"] = {
            key: metrics[key]
            for key in (
                "candidate_count", "retrieved_count", "best_relevance_score",
                "min_relevance_score",
            )
            if key in metrics
        }
        user_hash = hashlib.sha256(user.encode("utf-8")).hexdigest()[:16]
        logger.info(
            "ask status=%s latency_ms=%s question_chars=%s retrieved_count=%s "
            "candidate_count=%s best_relevance_score=%s abstain_reason=%s principal_id=%s",
            final.get("status", "error"), elapsed_ms, len(question),
            metrics.get("retrieved_count", 0), metrics.get("candidate_count", 0),
            metrics.get("best_relevance_score", 0.0), final.get("reason", ""), user_hash,
        )
        return final

    def ingest_path(self, path: Path, *, acl_labels: tuple[str, ...] = ("public",)) -> dict[str, int]:
        self._ensure_rag()
        store = self._ensure_store()
        labels = self.validate_acl_labels(acl_labels)
        root = path.resolve(strict=True)
        files = self._collect_files(root)
        summary = {"indexed": 0, "rejected": 0, "chunks": 0}
        if root.is_file() and root.suffix.casefold() not in SUPPORTED_EXTENSIONS:
            summary["rejected"] = 1
            return summary
        for file_path in files:
            doc_id = make_doc_id(os.path.normcase(str(file_path.resolve())))
            try:
                pages = load_document(
                    file_path,
                    max_file_bytes=self.settings.max_file_bytes,
                    timeout_seconds=self.settings.parser_timeout_seconds,
                    local_ocr=self.settings.local_ocr,
                    ocr_language=self.settings.ocr_language,
                    max_extracted_chars=self.settings.max_extracted_chars,
                    max_pdf_pages=self.settings.max_pdf_pages,
                )
                chunks = chunk_pages(
                    pages,
                    doc_id=doc_id,
                    source_name=sanitize_source_name(file_path.name),
                    acl_labels=labels,
                    chunk_size=self.settings.chunk_size,
                    chunk_overlap=self.settings.chunk_overlap,
                )
                if not chunks:
                    raise ValueError("document produced no chunks")
                if self.ollama is None:
                    raise RuntimeError("Local model client was not initialized")
                vectors: list[list[float]] = []
                for start in range(0, len(chunks), 32):
                    vectors.extend(self.ollama.embed([c.text for c in chunks[start:start + 32]]))
                count = store.upsert_document(chunks, vectors)
                summary["indexed"] += 1
                summary["chunks"] += count
                logger.info("ingest status=indexed doc_id=%s chunks=%s", doc_id, count)
            except Exception as exc:
                summary["rejected"] += 1
                logger.warning("ingest status=rejected doc_id=%s error_type=%s", doc_id, type(exc).__name__)
        return summary

    def delete_document(self, doc_id: str) -> int:
        if not _DOC_ID_RE.fullmatch(doc_id):
            raise ValueError("Document ID has an invalid format")
        deleted = self._ensure_store().delete_document(
            doc_id, allowed_groups=self.current_access_groups(),
        )
        logger.info("delete status=complete doc_id=%s chunk_count=%s", doc_id, deleted)
        return deleted

    def list_documents(self) -> list[dict[str, object]]:
        return self._ensure_store().documents(allowed_groups=self.current_access_groups())

    @staticmethod
    def validate_acl_labels(labels: tuple[str, ...]) -> tuple[str, ...]:
        if not labels or len(labels) > 32:
            raise ValueError("Provide between 1 and 32 ACL labels")
        normalized = {label.strip().casefold() for label in labels}
        if any(not _GROUP_RE.fullmatch(label) for label in normalized):
            raise ValueError("ACL labels contain an invalid value")
        return tuple(sorted(normalized))

    @staticmethod
    def _collect_files(root: Path) -> list[Path]:
        if root.is_file():
            return [root] if root.suffix.casefold() in SUPPORTED_EXTENSIONS else []
        files: list[Path] = []
        for candidate in root.rglob("*"):
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(root)
            except (OSError, ValueError):
                continue
            if resolved.is_file() and resolved.suffix.casefold() in SUPPORTED_EXTENSIONS:
                files.append(resolved)
        return sorted(set(files))
