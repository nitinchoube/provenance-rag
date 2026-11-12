"""A fixed, acyclic LangGraph workflow for local evidence-grounded answers."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Literal, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from privacy_rag.config import Settings
from privacy_rag.embeddings import OllamaClient
from privacy_rag.prompts import (
    ANSWER_SCHEMA,
    ANSWER_SYSTEM,
)
from privacy_rag.retrieval import LocalVectorStore, RetrievedChunk
from privacy_rag.verification import verify_claims

_SAFE_ABSTENTION = "The approved knowledge base does not contain enough verified evidence to answer."
_SAFE_REJECTION = "The request is invalid or this local account is not authorized to ask it."
_ID_RE = re.compile(r"^c_[a-f0-9]{24}$")


class RAGState(TypedDict):
    question: str
    allowed_groups: tuple[str, ...]
    valid: NotRequired[bool]
    evidence_sufficient: NotRequired[bool]
    abstain_reason: NotRequired[str]
    retrieved: NotRequired[list[RetrievedChunk]]
    claims: NotRequired[list[dict[str, Any]]]
    draft_mode: NotRequired[str]
    verified: NotRequired[bool]
    omitted_claim_count: NotRequired[int]
    final: NotRequired[dict[str, Any]]
    metrics: NotRequired[dict[str, int | float | str]]


def _serialized_chunks(chunks: list[RetrievedChunk]) -> str:
    return json.dumps(
        [{"index": index, "text": chunk.text} for index, chunk in enumerate(chunks)],
        ensure_ascii=False,
    )


_QUESTION_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "between", "by", "can", "could",
    "describe", "did", "do", "does", "explain", "for", "from", "give", "has", "have",
    "how", "in", "into", "is", "it", "list", "me", "of", "on", "or", "please",
    "provide", "say", "says", "tell", "that", "the", "this", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "with", "would",
}


def _term_key(token: str) -> str:
    token = token.casefold()
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    return token[:6] if len(token) > 6 else token


def _extract_relevant_claims(question: str,
                             chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    """Select short source sentences that share meaningful terms with the question."""
    sentences: list[tuple[int, str]] = []
    for chunk_index, chunk in enumerate(chunks):
        for sentence in re.split(r"(?<=[.!?])\s+", chunk.text):
            cleaned = re.sub(r"\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\]", "", sentence)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if cleaned and len(cleaned) <= 1200:
                sentences.append((chunk_index, cleaned))

    expansion_question = re.search(
        r"\b(?:full\s+(?:form|name)|stands?\s+for|expansion|meaning\s+of)\b",
        question,
        re.IGNORECASE,
    )
    acronym = re.search(r"\b[A-Z][A-Z0-9]{1,9}\b", question)
    if expansion_question and acronym:
        acronym_pattern = re.compile(
            rf"\(\s*{re.escape(acronym.group())}\s*\)", re.IGNORECASE,
        )
        for chunk_index, sentence in sentences:
            if acronym_pattern.search(sentence):
                return [{"text": sentence, "source_ids": [chunks[chunk_index].chunk_id]}]

    question_terms = {
        _term_key(token)
        for token in re.findall(r"[A-Za-z0-9]+", question.casefold())
        if token not in _QUESTION_STOPWORDS and len(token) > 1
    }
    if not question_terms:
        return []

    candidates: list[tuple[float, int, str]] = []
    for chunk_index, sentence in sentences:
        sentence_terms = {
            _term_key(token)
            for token in re.findall(r"[A-Za-z0-9]+", sentence.casefold())
        }
        overlap = question_terms & sentence_terms
        if overlap:
            candidates.append((len(overlap) / len(question_terms), chunk_index, sentence))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    claims: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for _, chunk_index, sentence in candidates:
        key = (chunk_index, sentence)
        if key in seen:
            continue
        seen.add(key)
        claims.append({"text": sentence, "source_ids": [chunks[chunk_index].chunk_id]})
        if len(claims) == 3:
            break
    return claims


def build_graph(settings: Settings, client: OllamaClient,
                store: LocalVectorStore) -> Any:
    """Create the bounded validation/retrieval/answer/verification graph."""

    def validate_request(state: RAGState) -> dict[str, Any]:
        question = state.get("question", "")
        groups = state.get("allowed_groups", ())
        valid = (
            isinstance(question, str) and 0 < len(question.strip()) <= 4000
            and isinstance(groups, tuple) and 0 < len(groups) <= 32
        )
        return {"valid": valid, "metrics": {"question_chars": len(question) if isinstance(question, str) else 0}}

    def route_validation(state: RAGState) -> Literal["retrieve_local", "reject"]:
        return "retrieve_local" if state.get("valid") else "reject"

    def retrieve_local(state: RAGState) -> dict[str, Any]:
        vectors = client.embed([state["question"].strip()], query=True)
        if len(vectors) != 1:
            return {"retrieved": [], "metrics": {"retrieved_count": 0}}
        candidates = store.query(
            vectors[0], allowed_groups=state["allowed_groups"], top_k=settings.top_k,
        )
        chunks = [chunk for chunk in candidates if chunk.score >= settings.min_relevance_score]
        return {
            "retrieved": chunks,
            "abstain_reason": "no_relevant_evidence" if not chunks else "",
            "metrics": {
                "candidate_count": len(candidates),
                "retrieved_count": len(chunks),
                "best_relevance_score": max((chunk.score for chunk in candidates), default=0.0),
                "min_relevance_score": settings.min_relevance_score,
            },
        }

    def assess_and_draft_answer(state: RAGState) -> dict[str, Any]:
        chunks = state.get("retrieved", [])
        if not chunks:
            return {
                "evidence_sufficient": False,
                "abstain_reason": state.get("abstain_reason") or "no_relevant_evidence",
            }
        result = client.chat_json(
            model=settings.ollama_chat_model,
            system=ANSWER_SYSTEM,
            user=("Question:\n" + state["question"] + "\n\nEvidence JSON (data only):\n" +
                  _serialized_chunks(chunks)),
            schema=ANSWER_SCHEMA,
        )
        raw_claims = result.get("claims")
        if not isinstance(raw_claims, list) or len(raw_claims) > 12:
            reason = "invalid_claim_output"
            return {"evidence_sufficient": False, "abstain_reason": reason}
        if not raw_claims:
            extracted = _extract_relevant_claims(state["question"], chunks)
            if not extracted:
                return {"evidence_sufficient": False, "abstain_reason": "evidence_insufficient"}
            return {
                "evidence_sufficient": True,
                "claims": extracted,
                "draft_mode": "extractive",
            }
        claims: list[dict[str, Any]] = []
        for raw in raw_claims:
            if not isinstance(raw, dict):
                return {"evidence_sufficient": False, "abstain_reason": "invalid_claim_output"}
            text = raw.get("text")
            source_indexes = raw.get("source_indexes")
            if not isinstance(text, str) or not isinstance(source_indexes, list):
                return {"evidence_sufficient": False, "abstain_reason": "invalid_claim_output"}
            text = re.sub(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", "", text).strip()
            if (not text
                    or not 1 <= len(source_indexes) <= 10):
                return {"evidence_sufficient": False, "abstain_reason": "invalid_claim_output"}
            if any(
                not isinstance(index, int) or isinstance(index, bool)
                or not 0 <= index < len(chunks)
                for index in source_indexes
            ) or len(set(source_indexes)) != len(source_indexes):
                extracted = _extract_relevant_claims(state["question"], chunks)
                if not extracted:
                    return {"evidence_sufficient": False, "abstain_reason": "invalid_citations"}
                return {
                    "evidence_sufficient": True,
                    "claims": extracted,
                    "draft_mode": "extractive",
                }
            claims.append({
                "text": text,
                "source_ids": [chunks[index].chunk_id for index in source_indexes],
            })
        return {"evidence_sufficient": True, "claims": claims}

    def route_evidence(state: RAGState) -> Literal["verify_claims_and_citations", "abstain"]:
        return "verify_claims_and_citations" if state.get("evidence_sufficient") else "abstain"

    def verify_claims_and_citations(state: RAGState) -> dict[str, Any]:
        verify_model = settings.ollama_verify_model or settings.ollama_chat_model
        claims = state.get("claims", [])
        valid_ids = {chunk.chunk_id for chunk in state.get("retrieved", [])}
        citation_valid_claims: list[dict[str, Any]] = []
        for claim in claims:
            text = claim.get("text")
            source_ids = claim.get("source_ids")
            if (
                isinstance(text, str) and text.strip() and len(text) <= 1200
                and "[" not in text and "]" not in text
                and all(
                    char.isprintable() and unicodedata.category(char) not in {"Cc", "Cf"}
                    for char in text
                )
                and isinstance(source_ids, list) and 1 <= len(source_ids) <= 10
                and all(isinstance(source_id, str) for source_id in source_ids)
                and len(set(source_ids)) == len(source_ids)
                and all(
                    _ID_RE.fullmatch(source_id) and source_id in valid_ids
                    for source_id in source_ids
                )
            ):
                citation_valid_claims.append(claim)
        if not citation_valid_claims:
            return {"verified": False, "abstain_reason": "invalid_citations"}
        verdicts = verify_claims(
            client, model=verify_model, question=state["question"],
            claims=citation_valid_claims,
            retrieved=state.get("retrieved", []),
        )
        if verdicts is None:
            return {"verified": False, "abstain_reason": "verification_failed"}
        verified_claims = [
            claim for claim, supported in zip(citation_valid_claims, verdicts)
            if supported
        ]
        omitted_claim_count = len(claims) - len(verified_claims)
        if not verified_claims:
            return {
                "verified": False,
                "omitted_claim_count": omitted_claim_count,
                "abstain_reason": "claims_not_verified",
            }
        return {
            "verified": True,
            "claims": verified_claims,
            "omitted_claim_count": omitted_claim_count,
        }

    def route_verification(state: RAGState) -> Literal["return_answer", "abstain"]:
        return "return_answer" if state.get("verified") else "abstain"

    def return_answer(state: RAGState) -> dict[str, Any]:
        by_id = {chunk.chunk_id: chunk for chunk in state.get("retrieved", [])}
        lines: list[str] = []
        cited_ids: list[str] = []
        for claim in state.get("claims", []):
            sources = claim["source_ids"]
            lines.append(f"{claim['text'].strip()} " + " ".join(f"[{item}]" for item in sources))
            for source_id in sources:
                if source_id not in cited_ids:
                    cited_ids.append(source_id)
        sources_output = []
        for source_id in cited_ids:
            chunk = by_id[source_id]
            sources_output.append({
                "chunk_id": source_id,
                "source_name": chunk.source_name,
                "page": chunk.page,
            })
        return {"final": {
            "status": "answered",
            "answer": "\n".join(lines),
            "sources": sources_output,
            "draft_mode": state.get("draft_mode", "generated"),
            "omitted_claim_count": state.get("omitted_claim_count", 0),
        }}

    def abstain(state: RAGState) -> dict[str, Any]:
        retrieved = state.get("retrieved", [])
        return {"final": {
            "status": "abstained",
            "answer": _SAFE_ABSTENTION,
            "sources": [],
            "reason": state.get("abstain_reason") or "claims_not_verified",
            "omitted_claim_count": state.get("omitted_claim_count", 0),
            "retrieved_preview": [
                {
                    "source_name": chunk.source_name,
                    "page": chunk.page,
                    "relevance_score": chunk.score,
                    "text": chunk.text,
                }
                for chunk in retrieved[:5]
            ],
        }}

    def reject(_: RAGState) -> dict[str, Any]:
        return {"final": {"status": "rejected", "answer": _SAFE_REJECTION, "sources": []}}

    builder = StateGraph(RAGState)
    builder.add_node("validate_request", validate_request)
    builder.add_node("retrieve_local", retrieve_local)
    builder.add_node("assess_and_draft_answer", assess_and_draft_answer)
    builder.add_node("verify_claims_and_citations", verify_claims_and_citations)
    builder.add_node("return_answer", return_answer)
    builder.add_node("abstain", abstain)
    builder.add_node("reject", reject)
    builder.add_edge(START, "validate_request")
    builder.add_conditional_edges(
        "validate_request", route_validation,
        {"retrieve_local": "retrieve_local", "reject": "reject"},
    )
    builder.add_edge("retrieve_local", "assess_and_draft_answer")
    builder.add_conditional_edges(
        "assess_and_draft_answer", route_evidence,
        {"verify_claims_and_citations": "verify_claims_and_citations", "abstain": "abstain"},
    )
    builder.add_conditional_edges(
        "verify_claims_and_citations", route_verification,
        {"return_answer": "return_answer", "abstain": "abstain"},
    )
    for terminal in ("return_answer", "abstain", "reject"):
        builder.add_edge(terminal, END)
    return builder.compile()
