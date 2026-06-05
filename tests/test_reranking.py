import unittest

from src.reranking import (
    boost_by_summary,
    filter_by_tags,
    query_word_set,
    rerank,
    summary_match_count,
)


class TestQueryWordSet(unittest.TestCase):
    def test_lowercases_and_drops_short(self):
        words = query_word_set("What is the Vacation policy?")
        self.assertIn("vacation", words)
        self.assertIn("policy", words)
        self.assertNotIn("a", words)
        self.assertIn("the", words)

    def test_empty_query(self):
        self.assertEqual(query_word_set(""), set())

    def test_handles_hyphens(self):
        words = query_word_set("starter-plan")
        self.assertIn("starter-plan", words)


class TestSummaryMatchCount(unittest.TestCase):
    def test_counts_overlap(self):
        self.assertEqual(summary_match_count("vacation policy", "vacation days policy"), 2)

    def test_empty_summary(self):
        self.assertEqual(summary_match_count("vacation policy", ""), 0)

    def test_no_overlap(self):
        self.assertEqual(summary_match_count("apple banana", "cherry date"), 0)

    def test_empty_query(self):
        self.assertEqual(summary_match_count("", "anything goes here"), 0)


class TestFilterByTags(unittest.TestCase):
    def test_filters_zero_overlap(self):
        def extractor(q):
            return ["vacation"]

        candidates = [
            {"text": "a", "tags": ["vacation", "policy"]},
            {"text": "b", "tags": ["pricing"]},
        ]
        result = filter_by_tags("vacation days", candidates, extractor)
        self.assertEqual([c["text"] for c in result], ["a"])

    def test_no_query_tags_returns_all(self):
        def extractor(q):
            return []

        candidates = [{"text": "a", "tags": []}, {"text": "b", "tags": []}]
        self.assertEqual(
            [c["text"] for c in filter_by_tags("a is the", candidates, extractor)],
            ["a", "b"],
        )

    def test_empty_candidates(self):
        self.assertEqual(filter_by_tags("anything", [], lambda q: []), [])


class TestBoostBySummary(unittest.TestCase):
    def test_reduces_distance_when_summary_matches(self):
        candidates = [
            {"text": "match", "distance": 1.0, "summary": "vacation policy summary"},
            {"text": "miss", "distance": 1.0, "summary": "completely unrelated topic"},
        ]
        result = boost_by_summary("vacation policy", candidates)
        self.assertLess(result[0]["distance"], result[1]["distance"])

    def test_empty_candidates(self):
        self.assertEqual(boost_by_summary("query", []), [])

    def test_clamped_to_non_negative(self):
        candidates = [
            {"text": "match", "distance": 0.01, "summary": "vacation policy summary"},
        ]
        result = boost_by_summary("vacation policy", candidates)
        self.assertGreaterEqual(result[0]["distance"], 0.0)


class TestRerankFunction(unittest.TestCase):
    def _make_candidates(self):
        return [
            {"text": "a", "distance": 0.1, "file_name": "f1.md"},
            {"text": "b", "distance": 0.5, "file_name": "f1.md"},
            {"text": "c", "distance": 0.3, "file_name": "f2.md"},
        ]

    def test_empty_input(self):
        self.assertEqual(rerank([], "query"), [])

    def test_preserves_count(self):
        result = rerank(self._make_candidates(), "query")
        self.assertEqual(len(result), 3)

    def test_returns_candidate_dicts(self):
        result = rerank(self._make_candidates(), "query")
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
        result = rerank(
            candidates, "query",
            importance_fn=lambda c: importances[c["text"]],
        )
        self.assertEqual(result[0]["text"], "b")

    def test_diversity_penalty_spreads_repeated_files(self):
        candidates = [
            {"text": "a1", "distance": 0.1, "file_name": "f1.md"},
            {"text": "a2", "distance": 0.2, "file_name": "f1.md"},
            {"text": "b1", "distance": 0.15, "file_name": "f2.md"},
        ]
        result = rerank(candidates, "query")
        files_in_order = [r["file_name"] for r in result]
        self.assertNotEqual(files_in_order[0], files_in_order[1])


if __name__ == "__main__":
    unittest.main()
