"""Bounded, allow-listed local document parsers."""

from __future__ import annotations

import csv
import io
import multiprocessing
from multiprocessing.connection import Connection
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import fitz

SupportedExtension = Literal[".pdf", ".txt", ".md", ".csv"]
SUPPORTED_EXTENSIONS = frozenset({".pdf", ".txt", ".md", ".csv"})


class DocumentLoadError(ValueError):
    """A document could not be parsed within the configured limits."""


@dataclass(frozen=True, slots=True)
class PageText:
    text: str
    page: int | None


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text.replace("\x00", " "))
    text = "".join(char for char in text if char in "\n\r\t" or char.isprintable())
    return re.sub(r"[ \t]+", " ", text).strip()


def _parse_child(path_text: str, extension: str, local_ocr: bool, ocr_language: str,
                 max_extracted_chars: int, max_pdf_pages: int) -> list[tuple[str, int | None]]:
    path = Path(path_text)
    if extension == ".pdf":
        pages: list[tuple[str, int | None]] = []
        with fitz.open(path) as pdf:
            if pdf.page_count > max_pdf_pages:
                raise DocumentLoadError("PDF exceeds the configured page limit")
            extracted_chars = 0
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text")
                if local_ocr and len(text.strip()) < 24:
                    text_page = page.get_textpage_ocr(full=True, language=ocr_language, dpi=180)
                    text = page.get_text("text", textpage=text_page)
                if text.strip():
                    extracted_chars += len(text)
                    if extracted_chars > max_extracted_chars:
                        raise DocumentLoadError("PDF exceeds the configured extraction limit")
                    pages.append((_normalize(text), page_number))
        return pages
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig", errors="strict")
    if len(text) > max_extracted_chars:
        raise DocumentLoadError("Document exceeds the configured extraction limit")
    if extension == ".csv":
        rows = csv.reader(io.StringIO(text, newline=""))
        text = "\n".join(" | ".join(cell.strip() for cell in row) for row in rows)
        if len(text) > max_extracted_chars:
            raise DocumentLoadError("CSV exceeds the configured extraction limit")
    cleaned = _normalize(text)
    return [(cleaned, None)] if cleaned else []


def _parse_worker(connection: Connection, path: str, extension: str,
                  local_ocr: bool, ocr_language: str,
                  max_extracted_chars: int, max_pdf_pages: int) -> None:
    try:
        connection.send((True, _parse_child(
            path, extension, local_ocr, ocr_language, max_extracted_chars, max_pdf_pages,
        )))
    except BaseException as exc:
        try:
            connection.send((False, type(exc).__name__))
        except (BrokenPipeError, OSError):
            pass
    finally:
        try:
            connection.close()
        except OSError:
            pass


def load_document(path: Path, *, max_file_bytes: int, timeout_seconds: int,
                  local_ocr: bool, ocr_language: str, max_extracted_chars: int,
                  max_pdf_pages: int) -> list[PageText]:
    """Parse one supported file in a killable subprocess with byte/time limits."""
    resolved = path.resolve(strict=True)
    extension = resolved.suffix.casefold()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentLoadError("unsupported file type")
    size = resolved.stat().st_size
    if size > max_file_bytes:
        raise DocumentLoadError("file exceeds the configured size limit")
    if size == 0:
        raise DocumentLoadError("file is empty")
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_parse_worker,
        args=(child_connection, str(resolved), extension, local_ocr, ocr_language,
              max_extracted_chars, max_pdf_pages),
        daemon=True,
    )
    process.start()
    child_connection.close()
    try:
        if not parent_connection.poll(timeout_seconds):
            process.terminate()
            process.join(timeout=2)
            raise DocumentLoadError("parser timed out")
        ok, result = parent_connection.recv()
        process.join(timeout=2)
        if not ok:
            raise DocumentLoadError(f"parser failed ({result})")
        pages = [PageText(text=body, page=page) for body, page in result if body.strip()]
        if not pages:
            raise DocumentLoadError("document contains no extractable text")
        return pages
    except (EOFError, OSError) as exc:
        raise DocumentLoadError("parser process failed") from exc
    finally:
        parent_connection.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)
