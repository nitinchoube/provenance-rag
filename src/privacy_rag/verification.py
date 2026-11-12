"""Deterministic citation validation and local claim support checks."""

from __future__ import annotations

import json
import unicodedata
from typing import Any

from privacy_rag.embeddings import OllamaClient
from privacy_rag.prompts import VERDICT_SCHEMA, VERIFIER_SYSTEM
from privacy_rag.retrieval import RetrievedChunk


def verify_claims(client: OllamaClient, *, model: str, question: str,
                  claims: list[dict[str, Any]],
                  retrieved: list[RetrievedChunk]) -> list[bool] | None:
    """Return per-claim support verdicts, or None for malformed verifier output."""
    if not claims or len(claims) > 40:
        return None
    by_id = {chunk.chunk_id: chunk for chunk in retrieved}
    verifier_claims: list[dict[str, Any]] = []
    for claim in claims:
        text = claim.get("text")
        source_ids = claim.get("source_ids")
        if not isinstance(text, str) or not text.strip() or len(text) > 1200:
            return None
        if any(not char.isprintable() or unicodedata.category(char) in {"Cc", "Cf"} for char in text):
            return None
        if not isinstance(source_ids, list) or not 1 <= len(source_ids) <= 10:
            return None
        if any(not isinstance(source_id, str) or source_id not in by_id for source_id in source_ids):
            return None
        if len(set(source_ids)) != len(source_ids):
            return None
        # The renderer owns citations. A model-written citation-like token is rejected.
        if "[" in text or "]" in text:
            return None
        verifier_claims.append({
            "index": len(verifier_claims),
            "claim": text.strip(),
            "cited_sources": [
                {"id": source_id, "text": by_id[source_id].text}
                for source_id in source_ids
            ],
        })
    response = client.chat_json(
        model=model,
        system=VERIFIER_SYSTEM,
        user="Question:\n" + question + "\n\nVerify each claim for source support and question relevance.\n" +
             json.dumps(verifier_claims, ensure_ascii=False),
        schema=VERDICT_SCHEMA,
    )
    verdicts = response.get("verdicts")
    if not isinstance(verdicts, list) or len(verdicts) != len(claims):
        return None
    normalized: dict[int, bool] = {}
    for verdict in verdicts:
        if not isinstance(verdict, dict):
            return None
        index, supported = verdict.get("index"), verdict.get("supported")
        if not isinstance(index, int) or isinstance(index, bool) or not isinstance(supported, bool):
            return None
        if index in normalized:
            return None
        normalized[index] = supported
    if set(normalized) != set(range(len(claims))):
        return None
    return [normalized[index] for index in range(len(claims))]
