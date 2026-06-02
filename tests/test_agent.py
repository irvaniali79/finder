import unittest
from unittest.mock import MagicMock
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


if __name__ == "__main__":
    unittest.main()
