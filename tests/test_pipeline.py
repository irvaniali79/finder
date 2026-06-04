import unittest
from unittest.mock import MagicMock
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocessor import (
    extract_tags,
    expand_query,
    detect_question_intent,
    combine_query_embeddings,
    STOPWORDS,
)
from pipeline import Pipeline


class TestExtractTags(unittest.TestCase):
    def test_empty_query(self):
        self.assertEqual(extract_tags(""), [])

    def test_filters_stopwords(self):
        tags = extract_tags("What is the vacation policy?")
        self.assertIn("vacation", tags)
        self.assertIn("policy", tags)
        self.assertNotIn("what", tags)
        self.assertNotIn("is", tags)
        self.assertNotIn("the", tags)

    def test_lowercases_input(self):
        tags = extract_tags("MATERNITY Leave")
        self.assertIn("maternity", tags)
        self.assertIn("leave", tags)

    def test_handles_hyphens_and_underscores(self):
        tags = extract_tags("ProCRM starter-plan price")
        self.assertIn("procrm", tags)
        self.assertIn("starter-plan", tags)

    def test_drops_single_char_words(self):
        tags = extract_tags("a b c leave")
        self.assertNotIn("a", tags)
        self.assertNotIn("b", tags)
        self.assertIn("leave", tags)


class TestDetectQuestionIntent(unittest.TestCase):
    def test_what_is(self):
        self.assertEqual(detect_question_intent("What is the leave policy?"), "what is")

    def test_how_many(self):
        self.assertEqual(detect_question_intent("How many vacation days?"), "how many")

    def test_no_prefix(self):
        self.assertEqual(detect_question_intent("vacation days"), "")

    def test_case_insensitive(self):
        self.assertEqual(detect_question_intent("HOW DO I request time off"), "how do")

    def test_tell_me_about(self):
        self.assertEqual(detect_question_intent("Tell me about maternity leave"), "tell me about")


class TestExpandQuery(unittest.TestCase):
    def test_returns_at_least_original(self):
        variants = expand_query("anything goes")
        self.assertIn("anything goes", variants)
        self.assertGreaterEqual(len(variants), 1)

    def test_empty_query(self):
        self.assertEqual(expand_query(""), [])

    def test_produces_keywords_variant(self):
        variants = expand_query("What is the vacation policy?")
        self.assertEqual(len(variants), 3)
        self.assertIn("vacation policy", variants)
        self.assertIn("what is vacation policy", variants)

    def test_no_expansion_when_only_stopwords(self):
        variants = expand_query("what is the")
        self.assertEqual(variants, ["what is the"])

    def test_max_three_variants(self):
        variants = expand_query("What is the company vacation policy for full time employees?")
        self.assertLessEqual(len(variants), 3)


class TestCombineQueryEmbeddings(unittest.TestCase):
    def test_centroid_of_variants(self):
        model = MagicMock()
        model.get_text_embedding.side_effect = [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        centroid = combine_query_embeddings(model, ["q1", "q2"])
        self.assertEqual(centroid.dtype, np.float32)
        np.testing.assert_allclose(centroid, [0.5, 0.5])

    def test_empty_variants(self):
        centroid = combine_query_embeddings(MagicMock(), [])
        self.assertEqual(centroid.shape, (0,))
        self.assertEqual(centroid.dtype, np.float32)


class TestPipelineQueryExpansion(unittest.TestCase):
    def _make_retriever(self, index=None):
        retriever = MagicMock()
        retriever.embed_model = MagicMock()
        retriever.embed_model.get_text_embedding.side_effect = lambda t: [0.0, 0.0]
        retriever.index = index or MagicMock()
        store = MagicMock()
        store.get_many.side_effect = lambda ids: [
            {"text": f"chunk {i}", "file_name": "a.md"} for i in ids
        ]
        retriever.chunk_store = store
        return retriever

    def test_v0_uses_single_query_embedding(self):
        retriever = self._make_retriever()
        retriever.index.search.return_value = (
            np.array([[0.1, 0.2]]), np.array([[0, 1]]),
        )
        p = Pipeline(retriever, variant="V0", top_k=2)
        p.retrieve("What is the leave policy?")
        self.assertEqual(retriever.embed_model.get_text_embedding.call_count, 1)

    def test_build_query_embedding_uses_preprocessor_when_expansion_enabled(self):
        retriever = self._make_retriever()
        p = Pipeline.__new__(Pipeline)
        p.retriever = retriever
        p.features = ["query_expansion"]
        p.top_k = 3
        p.top_n = 20
        emb = p._build_query_embedding("What is the leave policy?")
        self.assertGreaterEqual(
            retriever.embed_model.get_text_embedding.call_count, 2
        )
        self.assertEqual(emb.dtype, np.float32)

    def test_build_query_embedding_single_when_no_expansion(self):
        retriever = self._make_retriever()
        p = Pipeline.__new__(Pipeline)
        p.retriever = retriever
        p.features = []
        emb = p._build_query_embedding("vacation policy")
        self.assertEqual(retriever.embed_model.get_text_embedding.call_count, 1)
        self.assertEqual(emb.dtype, np.float32)

    def test_v3_runs_end_to_end(self):
        retriever = self._make_retriever()
        retriever.index.search.return_value = (
            np.array([[0.1, 0.2, 0.3]]),
            np.array([[0, 1, 2]]),
        )
        retriever.chunk_store.get_many.side_effect = lambda ids: [
            {"text": f"chunk {i}", "file_name": f"f{i}.md",
             "tags": ["leave"], "summary": f"summary {i}",
             "importance": 0.5} for i in ids
        ]
        p = Pipeline(retriever, variant="V3", top_k=3)
        results = p.retrieve("What is the leave policy?")
        self.assertEqual(len(results), 3)
        for text, dist, metadata in results:
            self.assertIsInstance(text, str)
            self.assertIsInstance(dist, float)
            self.assertIsInstance(metadata, dict)

    def test_v4_runs_rerank_stage(self):
        retriever = self._make_retriever()
        retriever.index.search.return_value = (
            np.array([[0.1, 0.4, 0.2, 0.3, 0.5]]),
            np.array([[0, 1, 2, 3, 4]]),
        )
        store = MagicMock()
        store.get_many.side_effect = lambda ids: [
            {
                "text": f"chunk {i}",
                "file_name": "a.md" if i % 2 == 0 else "b.md",
            }
            for i in ids
        ]
        retriever.chunk_store = store
        p = Pipeline(retriever, variant="V4", top_k=3, top_n=5)
        results = p.retrieve("query")

        self.assertEqual(len(results), 3)
        for text, dist, metadata in results:
            self.assertIsInstance(text, str)
            self.assertIsInstance(dist, float)
            self.assertIsInstance(metadata, dict)

    def test_v5_runs_rerank_with_importance(self):
        retriever = self._make_retriever()
        retriever.index.search.return_value = (
            np.array([[0.5, 0.1]]),
            np.array([[0, 1]]),
        )
        retriever.chunk_store.get_many.side_effect = lambda ids: [
            {"text": "low importance", "file_name": "a.md"},
            {"text": "high importance", "file_name": "b.md"},
        ]
        p = Pipeline(retriever, variant="V5", top_k=2, top_n=2)
        importances = {"low importance": 0.1, "high importance": 0.9}
        results = p.retrieve(
            "query",
            importance_fn=lambda c: importances.get(c["text"], 0.0),
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "high importance")


class TestRetrieveTopN(unittest.TestCase):
    def _make_retriever(self, n_chunks=5):
        retriever = MagicMock()
        retriever.embed_model = MagicMock()
        retriever.embed_model.get_text_embedding.return_value = [0.0, 0.0]
        retriever.index = MagicMock()
        retriever.index.search.return_value = (
            np.array([list(range(n_chunks))], dtype="float32"),
            np.array([list(range(n_chunks))], dtype="int64"),
        )
        store = MagicMock()
        store.count.return_value = n_chunks
        store.get_many.side_effect = lambda ids: [
            {"text": f"chunk {i}", "file_name": f"file{i % 2}.md"} for i in ids
        ]
        retriever.chunk_store = store
        return retriever

    def test_returns_n_candidates(self):
        retriever = self._make_retriever(n_chunks=5)
        p = Pipeline(retriever, variant="V0", top_n=20)
        candidates = p.retrieve_top_n("query", n=5)
        self.assertEqual(len(candidates), 5)

    def test_candidate_structure(self):
        retriever = self._make_retriever(n_chunks=3)
        p = Pipeline(retriever, variant="V0")
        candidates = p.retrieve_top_n("query", n=3)
        for c in candidates:
            self.assertIn("text", c)
            self.assertIn("distance", c)
            self.assertIn("file_name", c)
            self.assertIsInstance(c["distance"], float)

    def test_uses_default_top_n(self):
        retriever = self._make_retriever(n_chunks=10)
        p = Pipeline(retriever, variant="V0", top_n=10)
        candidates = p.retrieve_top_n("query")
        self.assertEqual(len(candidates), 10)

    def test_empty_index_returns_empty_list(self):
        retriever = MagicMock()
        retriever.embed_model = MagicMock()
        retriever.index = None
        retriever.chunk_store = None
        p = Pipeline(retriever, variant="V0")
        self.assertEqual(p.retrieve_top_n("query"), [])

    def test_handles_negative_indices(self):
        retriever = MagicMock()
        retriever.embed_model = MagicMock()
        retriever.embed_model.get_text_embedding.return_value = [0.0, 0.0]
        retriever.index = MagicMock()
        retriever.index.search.return_value = (
            np.array([[0.1, 0.2, 0.3]], dtype="float32"),
            np.array([[0, -1, 1]], dtype="int64"),
        )
        store = MagicMock()
        store.count.return_value = 3
        store.get_many.side_effect = lambda ids: [
            {"text": f"chunk {i}", "file_name": "a.md"} if i >= 0 else None
            for i in ids
        ]
        retriever.chunk_store = store
        p = Pipeline(retriever, variant="V0")
        candidates = p.retrieve_top_n("query", n=3)
        self.assertEqual(len(candidates), 2)

    def test_candidate_includes_metadata_fields(self):
        retriever = MagicMock()
        retriever.embed_model = MagicMock()
        retriever.embed_model.get_text_embedding.return_value = [0.0, 0.0]
        retriever.index = MagicMock()
        retriever.index.search.return_value = (
            np.array([[0.1, 0.2]], dtype="float32"),
            np.array([[0, 1]], dtype="int64"),
        )
        store = MagicMock()
        store.count.return_value = 2
        store.get_many.side_effect = lambda ids: [
            {
                "text": f"chunk {i}",
                "file_name": f"f{i}.md",
                "tags": ["leave", "policy"],
                "summary": f"summary {i}",
                "importance": 0.7,
            }
            for i in ids
        ]
        retriever.chunk_store = store
        p = Pipeline(retriever, variant="V0")
        candidates = p.retrieve_top_n("query", n=2)
        self.assertEqual(len(candidates), 2)
        for c in candidates:
            self.assertIn("tags", c)
            self.assertIn("summary", c)
            self.assertIn("importance", c)
        self.assertEqual(candidates[0]["tags"], ["leave", "policy"])
        self.assertEqual(candidates[0]["importance"], 0.7)


class TestRerank(unittest.TestCase):
    def _make_candidates(self):
        return [
            {"text": "a", "distance": 0.1, "file_name": "f1.md"},
            {"text": "b", "distance": 0.5, "file_name": "f1.md"},
            {"text": "c", "distance": 0.3, "file_name": "f2.md"},
        ]

    def test_empty_input(self):
        self.assertEqual(Pipeline.rerank([], "query"), [])

    def test_preserves_count(self):
        candidates = self._make_candidates()
        result = Pipeline.rerank(candidates, "query")
        self.assertEqual(len(result), len(candidates))

    def test_returns_candidate_dicts(self):
        candidates = self._make_candidates()
        result = Pipeline.rerank(candidates, "query")
        for r in result:
            self.assertIn("text", r)
            self.assertIn("distance", r)
            self.assertIn("file_name", r)

    def test_uses_importance_when_provided(self):
        candidates = [
            {"text": "a", "distance": 0.1, "file_name": "f1.md"},
            {"text": "b", "distance": 0.1, "file_name": "f2.md"},
        ]
        importances = {"a": 0.0, "b": 1.0}
        result = Pipeline.rerank(
            candidates, "query",
            importance_fn=lambda c: importances[c["text"]],
        )
        self.assertEqual(result[0]["text"], "b")

    def test_default_no_importance(self):
        candidates = self._make_candidates()
        result = Pipeline.rerank(candidates, "query")
        self.assertEqual(len(result), 3)

    def test_diversity_penalty_spreads_repeated_files(self):
        candidates = [
            {"text": "a1", "distance": 0.1, "file_name": "f1.md"},
            {"text": "a2", "distance": 0.2, "file_name": "f1.md"},
            {"text": "b1", "distance": 0.15, "file_name": "f2.md"},
        ]
        result = Pipeline.rerank(candidates, "query")
        files_in_order = [r["file_name"] for r in result]
        self.assertNotEqual(
            files_in_order[0], files_in_order[1],
            "Diversity penalty should separate same-file chunks in top positions",
        )


class TestRetrieveTopK(unittest.TestCase):
    def _make_retriever(self, n=6):
        retriever = MagicMock()
        retriever.embed_model = MagicMock()
        retriever.embed_model.get_text_embedding.return_value = [0.0, 0.0]
        retriever.index = MagicMock()
        retriever.index.search.return_value = (
            np.array([list(range(n))], dtype="float32"),
            np.array([list(range(n))], dtype="int64"),
        )
        store = MagicMock()
        store.count.return_value = n
        store.get_many.side_effect = lambda ids: [
            {"text": f"chunk {i}", "file_name": f"file{i % 2}.md"} for i in ids
        ]
        retriever.chunk_store = store
        return retriever

    def test_returns_k_results(self):
        retriever = self._make_retriever(n=6)
        p = Pipeline(retriever, variant="V4", top_k=3, top_n=6)
        results = p.retrieve_top_k("query", k=3)
        self.assertEqual(len(results), 3)
        for text, dist, metadata in results:
            self.assertIsInstance(text, str)
            self.assertIsInstance(dist, float)
            self.assertIsInstance(metadata, dict)

    def test_uses_default_top_k(self):
        retriever = self._make_retriever(n=10)
        p = Pipeline(retriever, variant="V4", top_k=3, top_n=10)
        results = p.retrieve_top_k("query")
        self.assertEqual(len(results), 3)

    def test_top_n_larger_than_k(self):
        retriever = self._make_retriever(n=20)
        p = Pipeline(retriever, variant="V4", top_k=3, top_n=20)
        results = p.retrieve_top_k("query", k=3)
        self.assertEqual(len(results), 3)

    def test_returns_tuple_format(self):
        retriever = self._make_retriever(n=4)
        p = Pipeline(retriever, variant="V4", top_k=2, top_n=4)
        results = p.retrieve_top_k("query", k=2)
        for item in results:
            self.assertEqual(len(item), 3)


class TestMetadataVariants(unittest.TestCase):
    def _make_retriever(self, chunks):
        retriever = MagicMock()
        retriever.embed_model = MagicMock()
        retriever.embed_model.get_text_embedding.return_value = [0.0, 0.0]
        retriever.index = MagicMock()
        store = MagicMock()
        store.count.return_value = len(chunks)
        store.get_many.side_effect = lambda ids: [chunks[i] for i in ids]
        retriever.chunk_store = store
        ids = list(range(len(chunks)))
        retriever.index.search.return_value = (
            np.array([[0.1 * (i + 1) for i in ids]], dtype="float32"),
            np.array([ids], dtype="int64"),
        )
        return retriever

    def test_v1_filters_chunks_with_no_tag_overlap(self):
        chunks = [
            {"text": "vacation policy info", "file_name": "leave.md",
             "tags": ["vacation", "policy"], "summary": "vacation summary",
             "importance": 0.5},
            {"text": "product pricing info", "file_name": "guide.md",
             "tags": ["pricing", "product"], "summary": "pricing summary",
             "importance": 0.5},
        ]
        retriever = self._make_retriever(chunks)
        p = Pipeline(retriever, variant="V1", top_k=5)
        results = p.retrieve("vacation days")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "vacation policy info")

    def test_v1_keeps_all_when_no_query_tags(self):
        chunks = [
            {"text": "alpha", "file_name": "a.md", "tags": [],
             "summary": "", "importance": 0.5},
            {"text": "beta", "file_name": "b.md", "tags": [],
             "summary": "", "importance": 0.5},
        ]
        retriever = self._make_retriever(chunks)
        p = Pipeline(retriever, variant="V1", top_k=5)
        results = p.retrieve("a is the")
        self.assertEqual(len(results), 2)

    def test_v2_summary_boost_reduces_distance(self):
        chunks = [
            {"text": "alpha", "file_name": "a.md",
             "tags": ["vacation", "policy"],
             "summary": "vacation policy summary", "importance": 0.5},
            {"text": "beta", "file_name": "b.md",
             "tags": ["vacation", "policy"],
             "summary": "completely unrelated topic", "importance": 0.5},
        ]
        retriever = self._make_retriever(chunks)
        p = Pipeline(retriever, variant="V2", top_k=5)
        results = p.retrieve("vacation policy")
        self.assertEqual(len(results), 2)
        texts = [r[0] for r in results]
        self.assertEqual(texts[0], "alpha")

    def test_v3_combines_tag_filter_and_summary_boost(self):
        chunks = [
            {"text": "vacation policy info", "file_name": "leave.md",
             "tags": ["vacation", "policy"], "summary": "vacation policy details",
             "importance": 0.5},
            {"text": "vacation pricing", "file_name": "guide.md",
             "tags": ["vacation", "pricing"], "summary": "pricing info",
             "importance": 0.5},
            {"text": "product info", "file_name": "product.md",
             "tags": ["product"], "summary": "product summary",
             "importance": 0.5},
        ]
        retriever = self._make_retriever(chunks)
        p = Pipeline(retriever, variant="V3", top_k=5)
        results = p.retrieve("vacation policy")
        texts = [r[0] for r in results]
        self.assertNotIn("product info", texts)
        self.assertEqual(texts[0], "vacation policy info")


if __name__ == "__main__":
    unittest.main()
