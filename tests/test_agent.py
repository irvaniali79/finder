import unittest
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent import QAAgent


class TestQAAgent(unittest.TestCase):
    def setUp(self):
        os.environ["OPENROUTER_API_KEY"] = "test_api_key"
        self.agent = QAAgent()

    def tearDown(self):
        if "OPENROUTER_API_KEY" in os.environ:
            del os.environ["OPENROUTER_API_KEY"]

    def test_generate_prompt_contains_context_and_query(self):
        chunks = [("company is open 9 AM to 6 PM.", 0.0), ("full-time employees receive 20 days.", 0.0)]
        prompt = self.agent.generate_prompt("What are your business hours?", chunks)
        self.assertIn("company is open 9 AM to 6 PM.", prompt)
        self.assertIn("full-time employees receive 20 days.", prompt)
        self.assertIn("What are your business hours?", prompt)

    @patch("agent.requests.Session.post")
    def test_answer_returns_llm_text(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "We are open 9 AM to 6 PM EST."}}]
        }
        mock_post.return_value = mock_response
        
        answer = self.agent.answer("business hours?", [("context", 0.0)])
        self.assertEqual(answer, "We are open 9 AM to 6 PM EST.")
        mock_post.assert_called_once()

    def test_generate_prompt_does_not_include_metadata(self):
        chunks = [("answer text", 0.0)]
        prompt = self.agent.generate_prompt("q", chunks)
        self.assertIn("answer text", prompt)

    @patch("agent.requests.Session.post")
    def test_answer_raises_on_parse_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.text = "Bad Gateway"
        mock_response.json.side_effect = ValueError("No JSON object could be decoded")
        mock_post.return_value = mock_response
        
        with self.assertRaises(RuntimeError) as context:
            self.agent.answer("business hours?", [("context", 0.0)])
        self.assertIn("Failed to parse LLM response", str(context.exception))

    def test_missing_api_key_raises_error(self):
        del os.environ["OPENROUTER_API_KEY"]
        with self.assertRaises(ValueError) as context:
            QAAgent()
        self.assertIn("OPENROUTER_API_KEY environment variable must be set", str(context.exception))

    def test_constructor_args_override_env_vars(self):
        with patch.dict(os.environ, {"OPENROUTER_MODEL": "env_model", "OPENROUTER_API_BASE": "http://env.com/api"}):
            agent = QAAgent(model_name="custom_model", api_base="http://custom.com/api")
            self.assertEqual(agent.model_name, "custom_model")
            self.assertEqual(agent.api_base, "http://custom.com/api")

    def test_env_vars_override_defaults(self):
        with patch.dict(os.environ, {"OPENROUTER_MODEL": "env_model", "OPENROUTER_API_BASE": "http://env.com/api"}):
            agent = QAAgent()
            self.assertEqual(agent.model_name, "env_model")
            self.assertEqual(agent.api_base, "http://env.com/api")

    def test_default_model_and_url(self):
        agent = QAAgent()
        self.assertEqual(agent.model_name, "openrouter/free")
        self.assertEqual(agent.api_base, "https://openrouter.ai/api/v1/chat/completions")

    def test_generate_prompt_accepts_legacy_two_tuples(self):
        chunks = [("answer text", 0.1)]
        prompt = self.agent.generate_prompt("q", chunks)
        self.assertIn("answer text", prompt)
        self.assertNotIn("[source:", prompt)

    def test_generate_prompt_includes_file_name_and_summary(self):
        chunks = [(
            "Full text of the chunk.",
            0.1,
            {
                "file_name": "leave.md",
                "summary": "Short summary.",
                "tags": ["vacation"],
                "importance": 0.5,
            },
        )]
        prompt = self.agent.generate_prompt("vacation days?", chunks)
        self.assertIn("[source: leave.md]", prompt)
        self.assertIn("summary: Short summary.", prompt)
        self.assertIn("Full text of the chunk.", prompt)

    def test_generate_prompt_omits_summary_when_equal_to_text(self):
        chunks = [(
            "Identical text.",
            0.1,
            {"file_name": "f.md", "summary": "Identical text.", "tags": []},
        )]
        prompt = self.agent.generate_prompt("q", chunks)
        self.assertNotIn("summary:", prompt)
        self.assertIn("[source: f.md]", prompt)

    def test_generate_prompt_handles_missing_metadata_keys(self):
        chunks = [("text only", 0.1, {})]
        prompt = self.agent.generate_prompt("q", chunks)
        self.assertIn("text only", prompt)
        self.assertNotIn("[source:", prompt)
        self.assertNotIn("summary:", prompt)


if __name__ == "__main__":
    unittest.main()