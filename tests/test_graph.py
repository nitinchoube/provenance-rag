from types import SimpleNamespace
import unittest

from privacy_rag.graph import build_graph
from privacy_rag.retrieval import RetrievedChunk


class FakeClient:
    def __init__(self, source_indexes: list[int] | None = None,
                 claim_text: str = "There are 1,010 cases.") -> None:
        self.chat_calls = 0
        self.source_indexes = source_indexes if source_indexes is not None else [2]
        self.claim_text = claim_text

    def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        return [[0.0]]

    def chat_json(self, *, model: str, system: str, user: str,
                  schema: dict[str, object]) -> dict[str, object]:
        self.chat_calls += 1
        if self.chat_calls == 1:
            return {"claims": [{
                "text": self.claim_text,
                "source_indexes": self.source_indexes,
            }]}
        return {"verdicts": [{"index": 0, "supported": True}]}


class FakeStore:
    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self.chunks = chunks or [RetrievedChunk(
            chunk_id="c_0123456789abcdef01234567",
            text="The LIDC-IDRI dataset includes 1,010 thorax CT cases with annotations.",
            score=0.7,
            doc_id="d_0123456789abcdef01234567",
            source_name="sample.pdf",
            page=4,
            acl_labels=("public",),
        )]

    def query(self, vector: list[float], *, allowed_groups: tuple[str, ...],
              top_k: int) -> list[RetrievedChunk]:
        return self.chunks[:top_k]


class InvalidCitationFallbackTests(unittest.TestCase):
    def test_acronym_expansion_fallback_prefers_explicit_definition(self) -> None:
        chunks = [
            RetrievedChunk(
                chunk_id="c_1123456789abcdef01234567",
                text="The LIDC-IDRI dataset includes 1,010 thorax CT cases with annotations.",
                score=0.67,
                doc_id="d_0123456789abcdef01234567",
                source_name="sample.pdf",
                page=4,
                acl_labels=("public",),
            ),
            RetrievedChunk(
                chunk_id="c_2123456789abcdef01234567",
                text=("The Lung Image Database Consortium (LIDC) and Image Database "
                      "Resource Initiative (IDRI), called LIDC-\nIDRI [4]."),
                score=0.62,
                doc_id="d_0123456789abcdef01234567",
                source_name="sample.pdf",
                page=4,
                acl_labels=("public",),
            ),
        ]
        client = FakeClient(source_indexes=[2])
        settings = SimpleNamespace(
            ollama_chat_model="granite4.1:3b",
            ollama_verify_model=None,
            top_k=5,
            min_relevance_score=0.6,
        )
        graph = build_graph(settings, client, FakeStore(chunks))

        result = graph.invoke({
            "question": "LIDC full form",
            "allowed_groups": ("public",),
        })["final"]

        self.assertEqual(result["status"], "answered")
        self.assertTrue(result["answer"].startswith("The Lung Image Database Consortium (LIDC)"))
        self.assertNotIn("1,010", result["answer"])
        self.assertEqual(result["sources"][0]["chunk_id"], "c_2123456789abcdef01234567")

    def test_out_of_range_draft_index_uses_verified_extractive_claim(self) -> None:
        client = FakeClient()
        settings = SimpleNamespace(
            ollama_chat_model="granite4.1:3b",
            ollama_verify_model=None,
            top_k=5,
            min_relevance_score=0.6,
        )
        graph = build_graph(settings, client, FakeStore())

        result = graph.invoke({
            "question": "How many thorax CT cases are in LIDC-IDRI?",
            "allowed_groups": ("public",),
        })["final"]

        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["draft_mode"], "extractive")
        self.assertEqual(result["sources"][0]["chunk_id"], "c_0123456789abcdef01234567")
        self.assertEqual(client.chat_calls, 2)

    def test_numeric_pdf_reference_marker_does_not_reject_claim(self) -> None:
        client = FakeClient(
            source_indexes=[0],
            claim_text="The Lung Image Database Consortium (LIDC) [4].",
        )
        settings = SimpleNamespace(
            ollama_chat_model="granite4.1:3b",
            ollama_verify_model=None,
            top_k=5,
            min_relevance_score=0.6,
        )
        graph = build_graph(settings, client, FakeStore())

        result = graph.invoke({
            "question": "What does LIDC stand for?",
            "allowed_groups": ("public",),
        })["final"]

        self.assertEqual(result["status"], "answered")
        self.assertIn("Lung Image Database Consortium (LIDC)", result["answer"])
        self.assertNotIn("[4]", result["answer"])
        self.assertEqual(client.chat_calls, 2)


if __name__ == "__main__":
    unittest.main()