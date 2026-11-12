"""Run a measured local evaluation over a JSONL golden set (no results are invented)."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from privacy_rag.app import RAGApplication  # noqa: E402
from privacy_rag.config import Settings  # noqa: E402
from privacy_rag.privacy import configure_logging  # noqa: E402


def _load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if not isinstance(item, dict) or not isinstance(item.get("question"), str):
                raise ValueError("Each golden row needs a question string")
            if not isinstance(item.get("should_answer"), bool):
                raise ValueError("Each golden row needs a boolean should_answer")
            expected = item.get("expected_chunk_ids", [])
            if not isinstance(expected, list) or any(not isinstance(value, str) for value in expected):
                raise ValueError("expected_chunk_ids must be a list of strings")
            cases.append(item)
    if not cases:
        raise ValueError("Golden set is empty")
    return cases


def evaluate(path: Path) -> dict[str, Any]:
    cases = _load_cases(path)
    latencies: list[float] = []
    answerable_hits = 0
    expected_citations = 0
    correct_expected_citations = 0
    abstentions_correct = 0
    settings = Settings.from_env()
    with RAGApplication(settings) as app:
        model_metadata = app.model_metadata()
        documents = app.list_documents()
        for case in cases:
            started = time.perf_counter()
            result = app.ask(case["question"])
            latencies.append((time.perf_counter() - started) * 1000)
            answered = result.get("status") == "answered"
            if case["should_answer"] and answered:
                answerable_hits += 1
            if not case["should_answer"] and not answered:
                abstentions_correct += 1
            expected = set(case.get("expected_chunk_ids", []))
            returned = {source.get("chunk_id") for source in result.get("sources", [])}
            expected_citations += len(expected)
            correct_expected_citations += len(expected & returned)
    ordered = sorted(latencies)
    p95_index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered) + 0.999) - 1))
    warm_latencies = sorted(latencies[1:])
    warm_p95_index = max(0, min(len(warm_latencies) - 1,
                                int(0.95 * len(warm_latencies) + 0.999) - 1)) if warm_latencies else 0
    positives = sum(1 for case in cases if case["should_answer"])
    negatives = len(cases) - positives
    storage_path = settings.vector_store_path
    index_storage_bytes = sum(
        item.stat().st_size for item in storage_path.rglob("*")
        if item.is_file()
    ) if storage_path.exists() else 0
    return {
        "cases": len(cases),
        "answerable_cases": positives,
        "answerable_hit_rate": answerable_hits / positives if positives else 0.0,
        "unanswerable_cases": negatives,
        "abstention_accuracy": abstentions_correct / negatives if negatives else 0.0,
        "expected_citation_recall": correct_expected_citations / expected_citations if expected_citations else 0.0,
        "latency_ms_median": statistics.median(latencies),
        "latency_ms_p95": ordered[p95_index],
        "first_question_latency_ms": latencies[0],
        "warm_latency_ms_median": statistics.median(warm_latencies) if warm_latencies else 0.0,
        "warm_latency_ms_p95": warm_latencies[warm_p95_index] if warm_latencies else 0.0,
        "run_metadata": {
            "models": model_metadata,
            "os": platform.platform(),
            "hardware": platform.processor() or platform.machine(),
            "document_count": len(documents),
            "indexed_chunk_count": sum(int(item["chunks"]) for item in documents),
            "index_storage_bytes": index_storage_bytes,
            "chunk_size_chars": settings.chunk_size,
            "chunk_overlap_chars": settings.chunk_overlap,
            "top_k": settings.top_k,
            "num_ctx": settings.ollama_num_ctx,
            "trial_count": len(cases),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, required=True, help="JSONL questions and expected citations")
    args = parser.parse_args()
    configure_logging()
    try:
        print(json.dumps(evaluate(args.golden), sort_keys=True, indent=2))
        return 0
    except Exception as exc:
        # The question row is private input; report only the exception class.
        print(f"Evaluation failed safely (error_type={type(exc).__name__}).", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
