"""Preparation must not turn a proposal into spending permission."""

import hashlib
import unittest

from tools.release470_plan import manifest


class PlanTests(unittest.TestCase):
    def test_plan_binds_exact_six_distinct_questions(self):
        result = manifest("offline-example")
        self.assertEqual(len(result["case_plan"]), 6)
        self.assertEqual(len(set(result["case_plan"].values())), 6)
        for case, text in result["questions"].items():
            self.assertEqual(
                result["case_plan"][case], hashlib.sha256(text.encode()).hexdigest()
            )
        self.assertNotEqual(
            result["case_plan"]["faq-1"], manifest("other")["case_plan"]["faq-1"]
        )

    def test_budget_proposals_remain_separate_and_unapproved(self):
        result = manifest("offline-example")
        cloud = result["cloudru"]
        self.assertEqual(cloud["calculated_micro_rub"], 322200)
        self.assertLess(cloud["calculated_micro_rub"], cloud["proposed_micro_rub_limit"])
        self.assertFalse(cloud["approved"])
        self.assertIsNone(result["index_rebuild"]["embedding_tokens"])
        self.assertIsNone(result["index_rebuild"]["micro_rub_limit"])
        self.assertFalse(result["index_rebuild"]["approved"])
        self.assertEqual(result["deepseek"]["approved_micro_cny_limit"], 3000000)

    def test_no_unresolved_or_path_like_run_id(self):
        for value in ("", "../run", "{run_id}", "x" * 49):
            with self.assertRaises(ValueError):
                manifest(value)


if __name__ == "__main__":
    unittest.main()
