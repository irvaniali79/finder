import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from retrieval import DocumentRetriever
from agent import QAAgent


class TestEvals(unittest.TestCase):
    def setUp(self):
        self.retriever = DocumentRetriever()
        self.retriever.setup()

    def test_retrieval_contains_business_hours(self):
        chunks = self.retriever.retrieve("What are the business hours for customer support?", top_k=3)
        self.assertTrue(any("9 AM to 6 PM" in c for c, _ in chunks))

    def test_retrieval_contains_vacation_days(self):
        chunks = self.retriever.retrieve("How many vacation days do full-time employees receive per year?", top_k=3)
        self.assertTrue(any("20 vacation days" in c for c, _ in chunks))

    def test_retrieval_contains_maternity_leave(self):
        chunks = self.retriever.retrieve("What is the maternity leave policy?", top_k=3)
        self.assertTrue(any("12 weeks paid" in c for c, _ in chunks))

    def test_retrieval_contains_procrm_starter_price(self):
        chunks = self.retriever.retrieve("What is the starting price for ProCRM?", top_k=3)
        self.assertTrue(any("Starter: $29/month" in c for c, _ in chunks))

    def test_retrieval_contains_support_contact(self):
        chunks = self.retriever.retrieve("How can I contact customer support?", top_k=3)
        self.assertTrue(any("support@company.com" in c for c, _ in chunks))

    @unittest.skipUnless(os.getenv("FINDER_RUN_LLM_EVALS"), "Skipping LLM-dependent eval; set FINDER_RUN_LLM_EVALS=1 to run")
    def test_llm_answers_match_expected(self):
        agent = QAAgent()
        cases = [
            ("What are the business hours for customer support?", "We are open Monday through Friday, 9 AM to 6 PM EST."),
            ("How many vacation days do full-time employees receive per year?", "Full-time employees receive 20 vacation days per year."),
            ("What is the maternity leave policy?", "Maternity leave: 12 weeks paid"),
            ("What is the starting price for ProCRM?", "Starter: $29/month (up to 5 users)"),
            ("How can I contact customer support?", "Email support@company.com or call 1-800-COMPANY. Support is available 24/7."),
        ]
        for question, expected in cases:
            with self.subTest(question=question):
                chunks = self.retriever.retrieve(question, top_k=3)
                answer = agent.answer(question, chunks).strip()
                self.assertEqual(answer, expected)


if __name__ == "__main__":
    unittest.main()
