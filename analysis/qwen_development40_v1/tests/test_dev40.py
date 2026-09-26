from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
RUNNER = ROOT / "scripts/run_dev40.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("qwen_dev40_test_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader
    spec.loader.exec_module(module)
    return module


RUN = load_runner()


class Development40Tests(unittest.TestCase):
    def test_plan_is_disabled_development_only_and_source_bound(self):
        plan = json.loads((ROOT / "configs/execution_plan.json").read_text())
        self.assertFalse(plan["execution_enabled"])
        self.assertEqual((plan["reports"], plan["passes"], plan["maximum_generations"]), (40, 1, 40))
        self.assertFalse(plan["validation_inference"] or plan["training"] or plan["bulk_extraction"])
        self.assertFalse(plan["organizer_labels_in_model_inputs"])
        self.assertEqual(set(plan["source_sha256"]), RUN.REQUIRED_SOURCES)
        for relative, expected in plan["source_sha256"].items():
            actual = hashlib.sha256((REPO / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, expected)

    def test_exact_model_prompt_and_limits_are_frozen(self):
        plan = json.loads((ROOT / "configs/execution_plan.json").read_text())
        self.assertEqual(plan["model_id"], "Qwen/Qwen3-14B")
        self.assertEqual(plan["model_revision"], "40c069824f4251a91eefaf281ebe4c544efd3e18")
        self.assertEqual(plan["prompt_sha256"], RUN.CONTROL_SHA)
        self.assertEqual(plan["definitions_sha256"], RUN.DEFINITIONS_SHA)
        self.assertEqual((plan["precision"], plan["attention"], plan["batch_size"]),
                         ("bfloat16", "sdpa", 1))
        self.assertEqual((plan["max_input_tokens"], plan["max_new_tokens"], plan["context_limit"]),
                         (8192, 2048, 40960))
        self.assertEqual(plan["stop_token_ids"], [151645, 151643])
        self.assertEqual((plan["automatic_retries"], plan["repair_generations"]), (0, 0))

    def test_validate_plan_rejects_wrong_hash(self):
        with self.assertRaises(RUN.GateError):
            RUN.validate_plan("0" * 64)

    def test_private_outputs_cannot_be_written_into_public_analysis(self):
        with self.assertRaises(RUN.GateError):
            RUN.private(REPO / "analysis/qwen_development40_v1/leak.json")

    def test_private_prepared_package_when_present(self):
        prepared = REPO / "state/runs/qwen-development40-v1-private/prepared-control-a"
        if not prepared.exists():
            self.skipTest("private prepared package is not part of the public checkout")
        plan_path = ROOT / "configs/execution_plan.json"
        plan = RUN.validate_plan(hashlib.sha256(plan_path.read_bytes()).hexdigest())
        rows, prompts, tokens = RUN.load_package(prepared, plan)
        self.assertEqual((len(rows), len(prompts), len(tokens)), (40, 40, 40))
        self.assertTrue(all(set(row) == {"case_index", "StudyInstanceUID", "Report", "split", "report_sha256"}
                            for row in rows))


if __name__ == "__main__":
    unittest.main()
