"""Privacy and safe logging helpers."""

from __future__ import annotations

import hashlib
import logging
import os
import re

_LABEL_RE = re.compile(r"[^A-Za-z0-9._ -]+")


def configure_logging() -> None:
    """Configure concise logs; callers must never pass content or exception traces."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def principal_groups(principal: str, configured: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    """Resolve the current OS account to groups; unknown users receive public only."""
    return configured.get(os.path.normcase(principal.strip()), ("public",))


def acl_metadata_key(label: str) -> str:
    """Map an ACL label to a safe deterministic scalar metadata key."""
    digest = hashlib.sha256(label.casefold().encode("utf-8")).hexdigest()[:24]
    return f"acl_{digest}"


def sanitize_source_name(name: str) -> str:
    """Keep only a basename-like display label for citation metadata."""
    cleaned = _LABEL_RE.sub("_", name.replace("\\", "/").split("/")[-1]).strip(" .")
    return cleaned[:180] or "document"


