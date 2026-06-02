import sys
import os
import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np
import faiss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from retrieval import DocumentRetriever
from agent import QAAgent


class TestEvalSuite(unittest.TestCase):
    """Evaluation suite based on EVALS.md questions."""

    EVAL_QUESTIONS = [
        "What are the business hours for customer support?",
        "How many vacation days do full-time employees receive per year?",
        "What is the maternity leave policy?",
        "What is the starting price for ProCRM?",
        "How can I contact customer support?",
    ]

    EXPECTED_ANSWERS = {
        "What are the business hours for customer support?": "We are open Monday through Friday, 9 AM to 6 PM EST.",
        "How many vacation days do full-time employees receive per year?": "Full-time employees receive 20 vacation days per year.",
        "What is the maternity leave policy?": "Maternity leave: 12 weeks paid",
        "What is the starting price for ProCRM?": "Starter: $29/month (up to 5 users)",
        "How can I contact customer support?": "Email support@company.com or call 1-800-COMPANY. Support is available 24/7.",
    }

    EXPECTED_KEYWORDS = {
        "What are the business hours for customer support?": ["monday", "friday", "9 am", "6 pm"],
        "How many vacation days do full-time employees receive per year?": ["20", "vacation", "days"],
        "What is the maternity leave policy?": ["maternity", "12 weeks"],
        "What is the starting price for ProCRM?": ["starter", "$29", "procrm"],
        "How can I contact customer support?": ["support@company.com", "1-800-company", "24/7"],
    }


class TestDocumentRetrieverEval(TestEvalSuite):
    def _create_mock_retriever(self, doc_texts):
        retriever = DocumentRetriever.__new__(DocumentRetriever)
        retriever.docs_path = "tests/fixtures"
        retriever.embed_model = MagicMock()
        retriever.index = MagicMock()
        
        mock_nodes = []
        for i, text in enumerate(doc_texts):
            node = MagicMock()
            node.text = text
            mock_nodes.append(node)
        
        retriever.chunks = mock_nodes
        return retriever

    def test_retriever_returns_eval_evidence(self):
        doc_texts = [
            "We are open Monday through Friday, 9 AM to 6 PM EST.",
            "Full-time employees receive 20 vacation days per year.",
            "Maternity leave: 12 weeks paid",
            "Starter: $29/month (up to 5 users)",
            "Email support@company.com or call 1-800-COMPANY. Support is available 24/7.",
        ]
        retriever = self._create_mock_retriever(doc_texts)
        retriever.embed_model.get_text_embedding.return_value = [0.0] * 384
        
        distances = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=np.float32)
        indices = np.array([[0, 1, 2, 3, 4]], dtype=np.int64)
        retriever.embed_model.get_text_embedding.side_effect = lambda t: [0.1] * 384
        retriever.index.search.return_value = (distances, indices)
        
        for question in self.EVAL_QUESTIONS:
            with self.subTest(question=question):
                chunks = retriever.retrieve(question, top_k=5)
                self.assertTrue(len(chunks) > 0, f"No chunks for: {question}")
                for chunk_text, distance in chunks:
                    self.assertIsInstance(chunk_text, str)
                    self.assertIsInstance(distance, float)

    def test_retriever_setup_pipeline(self):
        retriever = DocumentRetriever.__new__(DocumentRetriever)
        retriever.docs_path = "tests/fixtures"
        retriever.embed_model = MagicMock()
        retriever.index = MagicMock()
        retriever.chunks = []
        
        with patch.object(retriever, "load_documents", return_value=["d1"]) as load_mock, \
             patch.object(retriever, "chunk_documents", return_value=[MagicMock(text="n1")]) as chunk_mock, \
             patch.object(retriever, "create_embeddings", return_value=np.array([[0.5, 0.5], [0.6, 0.6]], dtype="float32")) as emb_mock, \
             patch.object(retriever, "build_index") as build_mock:
            retriever.setup()
            load_mock.assert_called_once()
            chunk_mock.assert_called_once_with(["d1"])
            build_mock.assert_called_once()
            self.assertEqual(retriever.index.ntotal, 2)


class TestQAAgentEval(TestEvalSuite):
    def test_agent_generate_prompt_includes_context(self):
        agent = QAAgent()
        chunks = [("We are open Monday through Friday, 9 AM to 6 PM EST.", 0.1)]
        prompt = agent.generate_prompt("What are the business hours?", chunks)
        self.assertIn("We are open Monday through Friday", prompt)
        self.assertIn("What are the business hours", prompt)

    @patch("agent.Ollama")
    def test_agent_answer_with_expected_responses(self, MockOllama):
        agent = QAAgent()
        expected = self.EXPECTED_ANSWERS[
            "What are the business hours for customer support?"
        ]
        response = MagicMock()
        response.text = expected
        agent.llm = MagicMock()
        agent.llm.complete.return_value = response
        
        actual = agent.answer(
            "What are the business hours for customer support?",
            [("We are open Monday through Friday, 9 AM to 6 PM EST.", 0.1)]
        )
        self.assertEqual(actual, expected)


class TestMainCLIEval(TestEvalSuite):
    @patch("main.input")
    @patch("main.DocumentRetriever")
    @patch("main.QAAgent")
    def test_main_prints_eval_answer(self, MockAgent, MockRetriever, mock_input):
        self._setup_cli_mocks(MockRetriever, MockAgent, "What are the business hours?")
        mock_input.side_effect = [
            "What are the business hours for customer support?",
            "exit"
        ]
        captured = []
        with patch("builtins.print", side_effect=lambda *args, **kwargs: captured.append(" ".join(map(str, args)))):
            from main import main
            main()
        self.assertTrue(any(
            "We are open Monday through Friday, 9 AM to 6 PM EST" in line
            for line in captured
        ))

    @patch("main.input")
    @patch("main.DocumentRetriever")
    @patch("main.QAAgent")
    def test_main_exits_on_quit(self, MockAgent, MockRetriever, mock_input):
        from main import main
        mock_input.side_effect = ["quit"]
        called = []
        with patch("builtins.print", side_effect=lambda *args, **kwargs: called.append(args[0] if args else "")):
            main()
        self.assertTrue(any("Goodbye" in (c or "") for c in called))

    @staticmethod
    def _setup_cli_mocks(MockRetriever, MockAgent, answer_text):
        retriever = MockRetriever.return_value
        retriever.retrieve.return_value = [(answer_text, 0.1)]
        retriever.setup.return_value = retriever
        agent = MockAgent.return_value
        agent.answer.return_value = answer_text


if __name__ == "__main__":
    unittest.main()
