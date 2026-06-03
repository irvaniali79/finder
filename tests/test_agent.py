import unittest
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent import QAAgent


class TestQAAgent(unittest.TestCase):
    def setUp(self):
        self.agent = QAAgent.__new__(QAAgent)
        self.agent.llm = MagicMock()

    def test_generate_prompt_contains_context_and_query(self):
        chunks = [("company is open 9 AM to 6 PM.", 0.0), ("full-time employees receive 20 days.", 0.0)]
        prompt = self.agent.generate_prompt("What are your business hours?", chunks)
        self.assertIn("company is open 9 AM to 6 PM.", prompt)
        self.assertIn("full-time employees receive 20 days.", prompt)
        self.assertIn("What are your business hours?", prompt)

    def test_answer_returns_llm_text(self):
        self.agent.llm.complete.return_value = MagicMock(text="We are open 9 AM to 6 PM EST.")
        answer = self.agent.answer("business hours?", [("context", 0.0)])
        self.assertEqual(answer, "We are open 9 AM to 6 PM EST.")
        self.agent.llm.complete.assert_called_once()

    def test_generate_prompt_does_not_include_metadata(self):
        chunks = [("answer text", 0.0)]
        prompt = self.agent.generate_prompt("q", chunks)
        self.assertIn("answer text", prompt)

    @patch("agent.Ollama")
    def test_default_model_and_url(self, MockOllama):
        agent = QAAgent()
        MockOllama.assert_called_once_with(
            model="llama3.2", base_url="http://localhost:11434", request_timeout=120.0
        )

    @patch("agent.Ollama")
    def test_env_vars_override_defaults(self, MockOllama):
        with patch.dict(os.environ, {"OLLAMA_MODEL": "llama3.1", "OLLAMA_BASE_URL": "http://ollama.example.com:11434"}):
            QAAgent()
        MockOllama.assert_called_once_with(
            model="llama3.1", base_url="http://ollama.example.com:11434", request_timeout=120.0
        )

    @patch("agent.Ollama")
    def test_constructor_args_override_env_vars(self, MockOllama):
        with patch.dict(os.environ, {"OLLAMA_MODEL": "llama3.1", "OLLAMA_BASE_URL": "http://ollama.example.com:11434"}):
            QAAgent(model_name="llama3.2", base_url="http://localhost:11434")
        MockOllama.assert_called_once_with(
            model="llama3.2", base_url="http://localhost:11434", request_timeout=120.0
        )


if __name__ == "__main__":
    unittest.main()
