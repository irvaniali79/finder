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

    def test_v3_still_blocked_until_f13(self):
        retriever = self._make_retriever()
        p = Pipeline(retriever, variant="V3")
        with self.assertRaises(NotImplementedError):
            p.retrieve("query")

    def test_v4_still_blocked_for_rerank(self):
        retriever = self._make_retriever()
        p = Pipeline(retriever, variant="V4")
        with self.assertRaises(NotImplementedError):
            p.retrieve("query")


if __name__ == "__main__":
    unittest.main()
