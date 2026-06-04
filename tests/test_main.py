import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.main import main


class TestMainCLI(unittest.TestCase):
    @patch("src.main.input", side_effect=["What are your business hours?", "exit"])
    @patch("src.main.build_pipeline")
    @patch("src.main.QAAgent")
    def test_main_cli_prints_answer(self, MockAgent, MockPipeline, mock_input):
        pipeline = MockPipeline.return_value
        pipeline.retrieve.return_value = [
            ("We are open Monday through Friday, 9 AM to 6 PM EST.", 0.1, {"file_name": "faq.md"})
        ]
        agent = MockAgent.return_value
        agent.answer.return_value = "We are open Monday through Friday, 9 AM to 6 PM EST."
        captured = []
        with patch("builtins.print", side_effect=lambda *args, **kwargs: captured.append(" ".join(map(str, args)))):
            main()
        self.assertTrue(any("We are open Monday through Friday, 9 AM to 6 PM EST." in line for line in captured))

    @patch("src.main.input", side_effect=["", "quit"])
    @patch("src.main.build_pipeline")
    @patch("src.main.QAAgent")
    def test_main_cli_exits_on_quit(self, MockAgent, MockPipeline, mock_input):
        with patch("builtins.print") as mock_print:
            main()
            called = [args[0] for args, _ in mock_print.call_args_list]
        self.assertTrue(any("Goodbye!" in (c or "") for c in called))


if __name__ == "__main__":
    unittest.main()
