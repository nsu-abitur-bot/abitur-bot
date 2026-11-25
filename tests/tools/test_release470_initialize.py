"""Changes to questions or currency caps must not initialize a live budget."""

import json
import unittest
from pathlib import Path

from tools.admission_usage import IndexingLimit
from tools.release470_initialize import validate_plan

MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "docs/reports/release-470-data/pilot-plan-approved-2026-11-21.json"
)


class InitializationTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_approved_manifest_matches_all_six_questions_and_policy(self):
        args, digest = validate_plan(self.plan)
        self.assertEqual(len(args["case_plan"]), 6)
        self.assertEqual(digest, self.plan["expected_policy_sha256"])

    def test_modified_question_is_rejected(self):
        self.plan["questions"]["rag-1"] += " changed"
        with self.assertRaises(IndexingLimit):
            validate_plan(self.plan)

    def test_higher_money_limit_is_rejected(self):
        self.plan["budget_initialization"]["cloudru"]["micro_rub_limit"] += 1
        with self.assertRaises(IndexingLimit):
            validate_plan(self.plan)

    def test_proposal_is_not_authorization(self):
        self.plan["status"] = "proposal_not_launch_authorization"
        with self.assertRaises(IndexingLimit):
            validate_plan(self.plan)


if __name__ == "__main__":
    unittest.main()
