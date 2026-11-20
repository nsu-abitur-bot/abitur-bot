"""Offline pilot manifest and cost calculation. Never initializes a live budget."""

import argparse
import hashlib
import json
import re

from tools.release470_cloudru import MODEL, cost_micro_rub

QUESTIONS = {
    "faq-1": "EVAL470-{run_id}: контрольный FAQ до правки",
    "faq-2": "EVAL470-{run_id}: контрольный FAQ после правки",
    "rag-1": "Какие профили бакалавриата есть на ФИТ?",
    "rag-2": "Какие языки программирования изучают на ПИиКН?",
    "tools-1": "Какие направления бакалавриата есть на ФИТ НГУ? Укажи коды.",
    "tools-2": "Какие проходные баллы прошлых лет есть для ФИТ?",
}


def manifest(run_id: str) -> dict:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,47}", run_id):
        raise ValueError("Use a public run identifier, 1..48 ASCII characters")
    questions = {k: v.format(run_id=run_id) for k, v in QUESTIONS.items()}
    return {
        "run_id": run_id,
        "status": "proposal_not_launch_authorization",
        "questions": questions,
        "case_plan": {
            k: hashlib.sha256(v.encode()).hexdigest() for k, v in questions.items()
        },
        "cloudru": {
            "model": MODEL,
            "dimensions": 2048,
            "rate_rub_per_million": "32.22",
            "proposed_embedding_limit": 10000,
            "calculated_micro_rub": cost_micro_rub(10000, 3222),
            "proposed_micro_rub_limit": 1000000,
            "approved": False,
        },
        "deepseek": {
            "model": "deepseek-flash",
            "thinking": False,
            "input_limit": 300000,
            "output_limit": 10000,
            "approved_micro_cny_limit": 3000000,
        },
        "index_rebuild": {
            "owner": "Marat",
            "embedding_tokens": None,
            "micro_rub_limit": None,
            "approved": False,
            "illustrations_only": [
                {"tokens": n, "micro_rub": cost_micro_rub(n, 3222)}
                for n in (100000, 500000, 1000000)
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    # ASCII JSON also works with the default Windows redirected-output encoding.
    print(json.dumps(manifest(args.run_id), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
