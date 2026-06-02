import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from main import main


class TestMainCLI(unittest.TestCase):
    @patch("main.input", side_effect=["What are your business hours?", "exit"])
    @patch("main.DocumentRetriever")
    @patch("main.QAAgent")
    def test_main_cli_prints_answer(self, MockAgent, MockRetriever, mock_input):
        retriever = MockRetriever.return_value
        retriever.retrieve.return_value = [("We are open Monday through Friday, 9 AM to 6 PM EST.", 0.1)]
        agent = MockAgent.return_value
        agent.answer.return_value = "We are open Monday through Friday, 9 AM to 6 PM EST."
        captured = []
        with patch("builtins.print", side_effect=lambda *args, **kwargs: captured.append(" ".join(map(str, args)))):
            main()
        self.assertTrue(any("We are open Monday through Friday, 9 AM to 6 PM EST." in line for line in captured))

    @patch("main.input", side_effect=["", "quit"])
    @patch("main.DocumentRetriever")
    @patch("main.QAAgent")
    def test_main_cli_exits_on_quit(self, MockAgent, MockRetriever, mock_input):
        with patch("builtins.print") as mock_print:
            main()
            called = [args[0] for args, _ in mock_print.call_args_list]
        self.assertTrue(any("Goodbye!" in (c or "") for c in called))


if __name__ == "__main__":
    unittest.main()
