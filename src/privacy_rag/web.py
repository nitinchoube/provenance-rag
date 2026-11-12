"""Loopback-only Streamlit interface for the local privacy-first RAG app."""

from __future__ import annotations

import hashlib
import logging
import math
import uuid
from pathlib import Path
from typing import Any

import streamlit as st

from privacy_rag.app import RAGApplication
from privacy_rag.config import Settings
from privacy_rag.embeddings import OllamaError
from privacy_rag.loaders import SUPPORTED_EXTENSIONS
from privacy_rag.privacy import configure_logging, sanitize_source_name

logger = logging.getLogger(__name__)
_MAX_FILES_PER_BATCH = 20
_MIB = 1024 * 1024


def _save_upload(uploaded: Any, root: Path, max_file_bytes: int) -> Path:
    """Save an upload under a sanitized, non-overwriting name inside the raw folder."""
    source_name = sanitize_source_name(str(uploaded.name))
    suffix = Path(source_name).suffix.casefold()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError("Choose a PDF, TXT, Markdown, or CSV file.")

    content = uploaded.getvalue()
    if not content:
        raise ValueError("The selected file is empty.")
    if len(content) > max_file_bytes:
        raise ValueError("The selected file exceeds the configured size limit.")

    root.mkdir(parents=True, exist_ok=True)
    stem = Path(source_name).stem or "document"
    digest = hashlib.sha256(content).hexdigest()[:12]
    candidates = [root / source_name, root / f"{stem}_{digest}{suffix}"]

    for attempt in range(5):
        if attempt >= len(candidates):
            candidates.append(root / f"{stem}_{digest}_{uuid.uuid4().hex[:8]}{suffix}")
        target = candidates[attempt]
        try:
            with target.open("xb") as handle:
                handle.write(content)
            return target
        except FileExistsError:
            try:
                if target.is_file() and not target.is_symlink() and target.read_bytes() == content:
                    return target
            except OSError:
                pass
    raise OSError("Could not create a unique local upload path")


def _load_settings() -> Settings:
    try:
        return Settings.from_env()
    except Exception as exc:
        logger.error("web status=configuration_failed error_type=%s", type(exc).__name__)
        st.error("Local configuration could not be loaded. Check the project .env file.")
        st.stop()
    raise RuntimeError("Streamlit stopped after a configuration error")


def _display_answer(result: dict[str, object]) -> None:
    status = result.get("status")
    if status == "answered":
        if result.get("draft_mode") == "extractive":
            st.success("Verified source excerpt")
        else:
            st.success("Verified answer")
        omitted = result.get("omitted_claim_count", 0)
        if isinstance(omitted, int) and omitted > 0:
            st.caption(f"{omitted} draft claim(s) were omitted because the sources did not support them.")
    elif status == "abstained":
        st.info("The knowledge base did not return enough verified evidence.")
        reason = result.get("reason")
        diagnostics = result.get("diagnostics", {})
        if reason == "no_relevant_evidence":
            st.caption("No retrieved chunks met the relevance threshold for this question.")
            if isinstance(diagnostics, dict):
                best = diagnostics.get("best_relevance_score")
                threshold = diagnostics.get("min_relevance_score")
                if isinstance(best, (int, float)) and isinstance(threshold, (int, float)):
                    st.caption(f"Best match: {best:.3f} · required: {threshold:.3f}")
        elif reason == "evidence_insufficient":
            st.caption("Retrieved chunks contained no claim that directly answered the question.")
        elif reason == "invalid_claim_output":
            st.caption("The local model returned a malformed answer draft, so no answer was shown.")
        elif reason == "invalid_citations":
            st.caption("The draft cited chunks outside the retrieved evidence, so no answer was shown.")
        elif reason == "claims_not_verified":
            st.caption("The retrieved text did not directly support any draft claim, so no answer was shown.")
        elif reason == "verification_failed":
            st.caption("The local verifier could not validate the complete draft, so no answer was shown.")
        if reason != "no_relevant_evidence" and isinstance(diagnostics, dict):
            best = diagnostics.get("best_relevance_score")
            threshold = diagnostics.get("min_relevance_score")
            if isinstance(best, (int, float)) and isinstance(threshold, (int, float)):
                st.caption(f"Best match: {best:.3f}; threshold: {threshold:.3f}")
    else:
        st.warning("The question could not be processed for this local account.")

    answer = result.get("answer")
    st.text(answer if isinstance(answer, str) else "No answer was returned.")

    previews = result.get("retrieved_preview", [])
    if status == "abstained" and isinstance(previews, list) and previews:
        with st.expander("Retrieved chunks (not verified)"):
            for item in previews:
                if not isinstance(item, dict):
                    continue
                source_name = item.get("source_name", "document")
                page = item.get("page")
                score = item.get("relevance_score")
                details = [str(source_name)]
                if isinstance(page, int) and page > 0:
                    details.append(f"page {page}")
                if isinstance(score, (int, float)):
                    details.append(f"match {score:.3f}")
                st.caption(" · ".join(details))
                chunk_text = item.get("text")
                if isinstance(chunk_text, str):
                    st.text(chunk_text)

    raw_sources = result.get("sources", [])
    if isinstance(raw_sources, list) and raw_sources:
        with st.expander("Sources"):
            for source in raw_sources:
                if not isinstance(source, dict):
                    continue
                chunk_id = source.get("chunk_id", "")
                source_name = source.get("source_name", "document")
                page = source.get("page")
                page_text = f" · page {page}" if isinstance(page, int) and page > 0 else ""
                st.write(f"{source_name}{page_text}")
                st.code(str(chunk_id), language=None)


def _render_ask(settings: Settings) -> None:
    st.subheader("Ask your documents")
    st.write("Answers are generated from locally indexed files and include verified chunk citations.")

    with st.form("ask_form"):
        question = st.text_area(
            "Question",
            placeholder="What does the policy say about…",
            max_chars=4000,
            height=130,
        )
        submitted = st.form_submit_button("Ask", type="primary", use_container_width=True)

    if submitted:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                with st.spinner("Searching local files and verifying cited evidence…"):
                    with RAGApplication(settings) as app:
                        st.session_state["last_answer"] = app.ask(question)
            except OllamaError as exc:
                logger.error("web operation=ask status=failed error=%s", str(exc))
                st.error(str(exc))
            except Exception as exc:
                logger.error("web operation=ask status=failed error_type=%s", type(exc).__name__)
                st.error("The local answer service is unavailable. Check Ollama and project configuration.")

    result = st.session_state.get("last_answer")
    if isinstance(result, dict):
        st.divider()
        _display_answer(result)
        if st.button("Clear answer", key="clear_answer"):
            del st.session_state["last_answer"]
            st.rerun()


def _render_ingest(settings: Settings) -> None:
    st.subheader("Add documents")
    st.write("Files are stored under `data/raw` and indexed by the local models.")
    limit_mib = max(1, math.ceil(settings.max_file_bytes / _MIB))

    with st.form("ingest_form"):
        uploads = st.file_uploader(
            "Choose files",
            type=["pdf", "txt", "md", "csv"],
            accept_multiple_files=True,
            max_upload_size=limit_mib,
            help=(
                f"Up to {settings.max_file_bytes / _MIB:.1f} MiB per file; "
                f"maximum {_MAX_FILES_PER_BATCH} files per batch."
            ),
        )
        groups_text = st.text_input(
            "Access groups", value="public",
            help="Comma-separated labels. The local OS account running this app determines which groups it can query.",
        )
        submitted = st.form_submit_button("Save and index", type="primary", use_container_width=True)

    if not submitted:
        return
    if not uploads:
        st.warning("Choose at least one file.")
        return
    if len(uploads) > _MAX_FILES_PER_BATCH:
        st.error(f"A batch can contain at most {_MAX_FILES_PER_BATCH} files.")
        return

    labels = tuple(label.strip() for label in groups_text.split(",") if label.strip())
    saved_paths: list[Path] = []
    failures: list[str] = []
    try:
        labels = RAGApplication.validate_acl_labels(labels)
        for upload in uploads:
            try:
                path = _save_upload(upload, settings.documents_path, settings.max_file_bytes)
                if path not in saved_paths:
                    saved_paths.append(path)
            except ValueError as exc:
                safe_name = sanitize_source_name(str(upload.name))
                failures.append(f"{safe_name}: {exc}")

        if not saved_paths:
            for failure in failures:
                st.warning(failure)
            return

        indexed = 0
        rejected = 0
        chunk_count = 0
        with st.spinner("Indexing files locally…"):
            with RAGApplication(settings) as app:
                for path in saved_paths:
                    summary = app.ingest_path(path, acl_labels=labels)
                    indexed += summary["indexed"]
                    rejected += summary["rejected"]
                    chunk_count += summary["chunks"]

        if indexed:
            st.success(f"Indexed {indexed} file(s) into {chunk_count} chunks.")
        if rejected:
            st.warning(f"{rejected} file(s) could not be parsed or indexed. The saved originals remain in data/raw.")
        for failure in failures:
            st.warning(failure)
    except ValueError as exc:
        st.error(str(exc))
    except OllamaError as exc:
        logger.error("web operation=ingest status=failed error=%s", str(exc))
        st.error(f"{exc} Uploaded originals remain in data/raw; indexing did not complete.")
    except Exception as exc:
        logger.error("web operation=ingest status=failed error_type=%s", type(exc).__name__)
        st.error("Indexing failed safely. Check local configuration, available storage, and Ollama status.")


def _render_knowledge_base(settings: Settings) -> None:
    st.subheader("Knowledge base")
    st.write("Only records available to the OS account running this app are shown.")
    try:
        with RAGApplication(settings) as app:
            documents = app.list_documents()
    except Exception as exc:
        logger.error("web operation=list_documents status=failed error_type=%s", type(exc).__name__)
        st.error("The local index could not be read. Check the vector store configuration.")
        return

    if not documents:
        st.info("No accessible documents are indexed yet. Add files to get started.")
        return

    rows: list[dict[str, str | int]] = []
    for item in documents:
        raw_chunks = item.get("chunks", 0)
        raw_labels = item.get("acl_labels", [])
        chunks = raw_chunks if isinstance(raw_chunks, int) else 0
        labels = raw_labels if isinstance(raw_labels, list) else []
        rows.append({
            "Document": str(item.get("source_name", "document")),
            "Chunks": chunks,
            "Access groups": ", ".join(str(label) for label in labels),
            "Document ID": str(item.get("doc_id", "")),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    options = {
        str(item.get("doc_id", "")): str(item.get("source_name", "document"))
        for item in documents
    }
    selected_id = st.selectbox(
        "Select a document to remove from the index",
        options=list(options),
        format_func=lambda doc_id: f"{options[doc_id]} · {doc_id}",
    )
    st.caption("Removing a record from the index does not delete its original file from data/raw.")
    if st.button("Remove selected document from index", type="secondary"):
        try:
            with RAGApplication(settings) as app:
                deleted = app.delete_document(selected_id)
            if deleted:
                st.success(f"Removed {deleted} chunk(s) from the local index.")
                st.rerun()
            st.warning("No accessible chunks were removed. Refresh the list and try again.")
        except Exception as exc:
            logger.error("web operation=delete status=failed error_type=%s", type(exc).__name__)
            st.error("The selected record could not be removed from the local index.")


def main() -> None:
    configure_logging()
    st.set_page_config(page_title="Private RAG", page_icon="⌂", layout="wide")
    if st.get_option("server.address") != "127.0.0.1":
        st.error("This application must listen on 127.0.0.1. Restore the project Streamlit server.address setting.")
        st.stop()

    settings = _load_settings()
    st.markdown(
        """
        <style>
        .stApp { background: #f5f7f4; }
        [data-testid="stSidebar"] { background: #eef2ed; }
        .block-container { max-width: 1120px; padding-top: 2rem; }
        .hero { background: linear-gradient(115deg, #102a25, #21574a); color: #f4faf5;
                border-radius: 18px; padding: 2rem 2.2rem; margin-bottom: 1.4rem; }
        .hero .eyebrow { color: #a8d7bd; font-size: .78rem; font-weight: 700;
                          letter-spacing: .14em; text-transform: uppercase; }
        .hero h1 { color: #f4faf5; margin: .45rem 0; font-size: 2.25rem; }
        .hero p { color: #d8e9df; margin: 0; font-size: 1rem; }
        div[data-testid="stForm"] { background: white; border: 1px solid #dce5dd;
                                      border-radius: 14px; padding: 1rem; }
        </style>
        <div class="hero">
          <div class="eyebrow">Private knowledge workspace</div>
          <h1>Ask your documents</h1>
          <p>Local parsing, retrieval, and answers with verified citations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## Private RAG")
        st.caption("This app is configured for this machine only.")
        page = st.radio(
            "Workspace", ["Ask", "Add documents", "Knowledge base"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption(f"Local model service · {settings.ollama_base_url}")
        st.caption(f"Maximum file size · {settings.max_file_bytes / _MIB:.1f} MiB")
        st.caption("Browser sessions use the OS account running Streamlit.")

    if page == "Ask":
        _render_ask(settings)
    elif page == "Add documents":
        _render_ingest(settings)
    else:
        _render_knowledge_base(settings)


if __name__ == "__main__":
    main()
