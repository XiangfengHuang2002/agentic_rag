import unittest

from src.api import _build_decision_data


class TestDecisionData(unittest.TestCase):
    def test_build_decision_data_marks_low_score_reason(self):
        data = _build_decision_data([
            {"content": "例子", "vector_sim": 0.3, "rerank_score": 0.1}
        ])

        self.assertFalse(data["need_rag"])
        self.assertEqual(data["score"], 0.3)
        self.assertIn("低于阈值", data["reason"])

    def test_build_decision_data_marks_pass_reason(self):
        data = _build_decision_data([
            {"content": "例子", "vector_sim": 0.9, "rerank_score": 0.6}
        ])

        self.assertTrue(data["need_rag"])
        self.assertEqual(data["score"], 0.9)
        self.assertIn("高于阈值", data["reason"])

    def test_build_decision_data_handles_empty_chunks(self):
        data = _build_decision_data([])

        self.assertFalse(data["need_rag"])
        self.assertEqual(data["score"], 0.0)
        self.assertIn("未检索到任何背景片段", data["reason"])


if __name__ == "__main__":
    unittest.main()
