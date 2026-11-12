"""Typed application configuration with fail-closed privacy validation."""

from __future__ import annotations

import ipaddress
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit, urlunsplit

_GROUP_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,63}$")


class ConfigurationError(ValueError):
    """Raised when settings are invalid or weaken a privacy invariant."""


def _read_dotenv(path: Path) -> dict[str, str]:
    """Read simple KEY=VALUE entries without shell expansion or interpolation."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def _bool(values: Mapping[str, str], key: str, default: bool) -> bool:
    raw = values.get(key)
    if raw is None:
        return default
    normalized = raw.strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ConfigurationError(f"{key} must be true or false")


def _int(values: Mapping[str, str], key: str, default: int) -> int:
    try:
        return int(values.get(key, str(default)))
    except ValueError as exc:
        raise ConfigurationError(f"{key} must be an integer") from exc


def _float(values: Mapping[str, str], key: str, default: float) -> float:
    try:
        return float(values.get(key, str(default)))
    except ValueError as exc:
        raise ConfigurationError(f"{key} must be a number") from exc


def _parse_principals(raw: str) -> dict[str, tuple[str, ...]]:
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigurationError("ACL_PRINCIPALS_JSON must be a JSON object") from exc
    if not isinstance(decoded, dict):
        raise ConfigurationError("ACL_PRINCIPALS_JSON must be a JSON object")
    result: dict[str, tuple[str, ...]] = {}
    for principal, groups in decoded.items():
        if not isinstance(principal, str) or not principal.strip():
            raise ConfigurationError("ACL principal names must be non-empty strings")
        principal_key = os.path.normcase(principal.strip())
        if principal_key in result:
            raise ConfigurationError("ACL principal names must be unique ignoring case")
        if not isinstance(groups, list) or len(groups) > 32:
            raise ConfigurationError("Each ACL principal must map to a list of at most 32 groups")
        normalized: set[str] = set()
        for group in groups:
            if not isinstance(group, str) or not _GROUP_RE.fullmatch(group):
                raise ConfigurationError("ACL group labels contain an invalid value")
            normalized.add(group.casefold())
        normalized.add("public")
        if len(normalized) > 32:
            raise ConfigurationError("Each ACL principal may have at most 32 groups including public")
        result[principal_key] = tuple(sorted(normalized))
    return result


def _validate_loopback_url(raw: str) -> str:
    parsed = urlsplit(raw.strip())
    if (parsed.scheme != "http" or not parsed.hostname
            or parsed.username is not None or parsed.password is not None):
        raise ConfigurationError("OLLAMA_BASE_URL must be a plain HTTP loopback URL")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ConfigurationError("OLLAMA_BASE_URL must not include a path, query, or fragment")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ConfigurationError("OLLAMA_BASE_URL port is invalid") from exc
    port = 11434 if port is None else port
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        if parsed.hostname.casefold() != "localhost":
            raise ConfigurationError("OLLAMA_BASE_URL host must be a loopback address")
        host = "127.0.0.1"
    else:
        if not address.is_loopback:
            raise ConfigurationError("OLLAMA_BASE_URL host must be a loopback address")
        host = f"[{address.compressed}]" if address.version == 6 else address.compressed
    if not 1 <= port <= 65535:
        raise ConfigurationError("OLLAMA_BASE_URL port is invalid")
    return urlunsplit(("http", f"{host}:{port}", "", "", "")).rstrip("/")


@dataclass(frozen=True, slots=True)
class Settings:
    ollama_base_url: str
    ollama_chat_model: str
    ollama_verify_model: str | None
    ollama_embed_model: str
    ollama_num_ctx: int
    ollama_keep_alive: str
    ollama_timeout_seconds: int
    llm_temperature: float
    vector_store: str
    vector_store_path: Path
    documents_path: Path
    chunk_size: int
    chunk_overlap: int
    top_k: int
    min_relevance_score: float
    max_file_bytes: int
    max_extracted_chars: int
    max_pdf_pages: int
    parser_timeout_seconds: int
    local_ocr: bool
    ocr_language: str
    acl_principals: dict[str, tuple[str, ...]]

    @classmethod
    def from_env(cls, dotenv_path: Path | None = None) -> "Settings":
        """Load the project .env first, then let process environment override it."""
        values = _read_dotenv(dotenv_path or Path.cwd() / ".env")
        values.update(os.environ)
        for tracing_key in ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING"):
            if _bool(values, tracing_key, False):
                raise ConfigurationError("Framework tracing is disabled by the privacy policy")
        if _bool(values, "ALLOW_WEB_SEARCH", False):
            raise ConfigurationError("Web search is disabled by the privacy policy")
        if _bool(values, "ALLOW_EXTERNAL_LLM", False):
            raise ConfigurationError("External LLM providers are disabled by the privacy policy")
        if _bool(values, "LOG_PROMPTS", False):
            raise ConfigurationError("Prompt and content logging is disabled by the privacy policy")

        embed_model = values.get("OLLAMA_EMBED_MODEL", "nomic-embed-text").strip()
        if embed_model != "nomic-embed-text":
            raise ConfigurationError("OLLAMA_EMBED_MODEL is fixed to nomic-embed-text")
        vector_store = values.get("VECTOR_STORE", "chroma").strip().casefold()
        if vector_store != "chroma":
            raise ConfigurationError("VECTOR_STORE is fixed to the local Chroma backend")
        chat_model = values.get("OLLAMA_CHAT_MODEL", "").strip()
        if not chat_model or "cloud" in chat_model.casefold():
            raise ConfigurationError("OLLAMA_CHAT_MODEL must name an installed local model")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}", chat_model):
            raise ConfigurationError("OLLAMA_CHAT_MODEL contains an invalid model tag")
        verify_model = values.get("OLLAMA_VERIFY_MODEL", "").strip() or None
        if verify_model and (
            "cloud" in verify_model.casefold()
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}", verify_model)
        ):
            raise ConfigurationError("OLLAMA_VERIFY_MODEL must name an installed local model")
        num_ctx = _int(values, "OLLAMA_NUM_CTX", 8192)
        timeout = _int(values, "OLLAMA_TIMEOUT_SECONDS", 180)
        chunk_size = _int(values, "CHUNK_SIZE", 800)
        chunk_overlap = _int(values, "CHUNK_OVERLAP", 120)
        top_k = _int(values, "TOP_K", 5)
        max_file_bytes = _int(values, "MAX_FILE_BYTES", 25 * 1024 * 1024)
        max_extracted_chars = _int(values, "MAX_EXTRACTED_CHARS", 5_000_000)
        max_pdf_pages = _int(values, "MAX_PDF_PAGES", 1000)
        parser_timeout = _int(values, "PARSER_TIMEOUT_SECONDS", 30)
        threshold = _float(values, "MIN_RELEVANCE_SCORE", 0.65)
        temperature = _float(values, "LLM_TEMPERATURE", 0.0)
        if num_ctx < 1024 or timeout < 1 or top_k < 1:
            raise ConfigurationError("Context size, timeout, and TOP_K must be positive")
        if chunk_size < 100 or not 0 <= chunk_overlap < chunk_size:
            raise ConfigurationError("Chunk size must be >= 100 and overlap must be smaller")
        if min(max_file_bytes, max_extracted_chars, max_pdf_pages, parser_timeout) < 1:
            raise ConfigurationError("File, extraction, page, and parser timeout limits must be positive")
        if not 0 <= threshold <= 1 or temperature != 0:
            raise ConfigurationError("Relevance threshold must be in [0, 1] and temperature must be 0")
        if not values.get("OLLAMA_KEEP_ALIVE", "30m").strip():
            raise ConfigurationError("OLLAMA_KEEP_ALIVE must not be empty")
        return cls(
            ollama_base_url=_validate_loopback_url(values.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")),
            ollama_chat_model=chat_model,
            ollama_verify_model=verify_model,
            ollama_embed_model=embed_model,
            ollama_num_ctx=num_ctx,
            ollama_keep_alive=values.get("OLLAMA_KEEP_ALIVE", "30m").strip(),
            ollama_timeout_seconds=timeout,
            llm_temperature=temperature,
            vector_store=vector_store,
            vector_store_path=Path(values.get("VECTOR_STORE_PATH", "./storage/vectorstore")),
            documents_path=Path(values.get("DOCUMENTS_PATH", "./data/raw")),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            min_relevance_score=threshold,
            max_file_bytes=max_file_bytes,
            max_extracted_chars=max_extracted_chars,
            max_pdf_pages=max_pdf_pages,
            parser_timeout_seconds=parser_timeout,
            local_ocr=_bool(values, "LOCAL_OCR", True),
            ocr_language=values.get("OCR_LANGUAGE", "eng").strip() or "eng",
            acl_principals=_parse_principals(values.get("ACL_PRINCIPALS_JSON", "{}")),
        )
