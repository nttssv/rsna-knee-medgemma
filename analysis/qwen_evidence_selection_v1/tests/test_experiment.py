from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import experiment_core as exp


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


V2 = load_module("evidence_test_v2_core", REPO / "analysis/report_labeling_llm_v2/scripts/core.py")


class EvidenceSelectionTests(unittest.TestCase):
    def test_control_and_definitions_are_exact_frozen_sources(self):
        a = ROOT / "prompts/control_A.txt"
        defs = ROOT / "prompts/target_definitions.txt"
        self.assertEqual(exp.sha_file(a), "86b3fdf3adffff5e74b3b94192580c5d771711d441d682ecba88cc64bb344a68")
        self.assertEqual(exp.sha_file(defs), "dab72cb29da5105bc6f02336b50c3488028d92b0cf1b1bfe78f1038f6fd1264b")
        self.assertEqual(a.read_bytes(), (REPO / "analysis/qwen_report_extraction_v1/prompts/control.txt").read_bytes())

    def test_candidate_is_control_plus_one_inserted_instruction(self):
        a = (ROOT / "prompts/control_A.txt").read_text()
        b = (ROOT / "prompts/candidate_B.txt").read_text()
        addition = (ROOT / "prompts/evidence_selection_B.txt").read_text().rstrip("\n")
        marker = "Do not include explanations, markdown fences, reasoning traces, identifiers or any keys other than the required JSON schema."
        self.assertEqual(b, a.replace(marker, addition + "\n" + marker))
        self.assertNotIn("Report 1", b)
        self.assertNotIn("[DATE]", b)

    def test_candidate_guidance_covers_target_scope_negation_conflict_and_quote(self):
        b=(ROOT/"prompts/evidence_selection_B.txt").read_text().lower()
        for required in ("entire report", "exact target structure, tissue, compartment and time",
                         "every target it explicitly names", "explicit negative", "not_mentioned",
                         "adjacent normal anatomy", "different tissues or structures",
                         "one continuous exact passage", "do not join separate passages"):
            self.assertIn(required,b)

    def test_public_baseline_counts_are_pinned_and_unmodified(self):
        baseline=json.loads((ROOT/"configs/baseline_snapshot.json").read_text())
        self.assertEqual(baseline["reference_commit"],"2f6604bb68348dd275554e66f539869763a1bff0")
        self.assertEqual((baseline["raw_proposal_different_cells"],
                          baseline["primary_parsed_row_different_cells"],
                          baseline["raw_only_differences"]),(20,18,2))
        public=REPO/baseline["public_snapshot"]
        for name,expected in baseline["public_snapshot_files_sha256"].items():
            self.assertEqual(exp.sha_file(public/name),expected)

    def test_prompt_wiring_preserves_original_report_exactly(self):
        report = "REPORTE\nHallazgos: ligamento normal.\n"
        for v in ("A", "B"):
            built = exp.build_prompt(report, v)
            self.assertIn(json.dumps(report, ensure_ascii=False), built)
            self.assertTrue(built.endswith(json.dumps(report, ensure_ascii=False)))
            self.assertIn("TARGET DEFINITIONS:\n", built)
        self.assertEqual(exp.build_prompt(report, "A"),
            (ROOT / "prompts/control_A.txt").read_text() + "\nTARGET DEFINITIONS:\n" +
            (ROOT / "prompts/target_definitions.txt").read_text() +
            "\nREPORT_JSON_STRING:\n" + json.dumps(report, ensure_ascii=False))

    def test_same_frozen_model_sampling_parser_and_execution_lock(self):
        cfg = json.loads((ROOT / "configs/experiment.json").read_text())
        self.assertEqual(cfg["revision"], "40c069824f4251a91eefaf281ebe4c544efd3e18")
        self.assertEqual(cfg["run_order"], ["A1", "B1", "B2", "A2"])
        self.assertEqual(cfg["operator_block_mapping"], {"A1":"control-1","B1":"candidate-1","B2":"candidate-2","A2":"control-2"})
        self.assertEqual(cfg["maximum_generations"], 20)
        self.assertEqual((cfg["max_input_tokens"], cfg["max_new_tokens"], cfg["context_limit"]), (8192,2048,40960))
        self.assertEqual(cfg["stop_token_ids"], [151645,151643])
        self.assertFalse(cfg["execution_enabled"])
        self.assertFalse(cfg["training"] or cfg["validation_inference"] or cfg["bulk_extraction"])
        self.assertEqual(cfg["run_order"],["A1","B1","B2","A2"])

    def test_input_integrity_fails_closed(self):
        split = [{"StudyInstanceUID":f"s{i}","split":"development" if i<40 else "validation"} for i in range(58)]
        rows=[]
        for i in range(5):
            report=f"Synthetic report {i}."
            rows.append({"StudyInstanceUID":f"s{i}","Report":report,"split":"development","language":"synthetic",
                         "report_sha256":hashlib.sha256(report.encode()).hexdigest()})
        exp.validate_inputs(rows,split)
        with self.assertRaises(ValueError): exp.validate_inputs(rows[:4],split)
        with self.assertRaises(ValueError): exp.validate_inputs(rows+[rows[0]],split)
        bad=[dict(rows[0],organizer_labels={})]+rows[1:]
        with self.assertRaises(ValueError): exp.validate_inputs(bad,split)
        bad=[dict(rows[0],report_sha256="0"*64)]+rows[1:]
        with self.assertRaises(ValueError): exp.validate_inputs(bad,split)

    def test_source_span_is_exact_and_contiguous(self):
        report="ACL is intact.\nMCL is intact."
        quote="MCL is intact."
        start=report.index(quote)
        self.assertTrue(exp.verify_source_span(report,quote,start,start+len(quote)))
        self.assertFalse(exp.verify_source_span(report,"ACL is intact. MCL is intact.",0,len(report)))

    def test_existing_v2_parser_accepts_unchanged_12_condition_schema(self):
        value={name:{"label":"not_mentioned","evidence_text":"","confidence":0}
               for name in V2.LABELS}
        parsed=V2.validate_response(json.dumps(value), "A synthetic report with no target findings.")
        rows=parsed["rows"]
        self.assertEqual(parsed["parser_status"],"valid_json")
        self.assertEqual(len(rows),12)
        self.assertTrue(all(row["status"] == "valid" for row in rows))
        self.assertEqual([row["condition"] for row in rows],list(V2.LABELS))

    def test_repeated_quote_stays_technical_failure(self):
        report="ACL is intact. Elsewhere, ACL is intact."
        match=V2.match_evidence(report,"ACL is intact.")
        self.assertEqual(match["status"],"ambiguous_evidence_error")

    def test_comparison_separates_raw_proposals_from_accepted_rows(self):
        left=[]; right=[]
        for i in range(5):
            for j in range(12):
                common={"case_index":i,"condition":f"C{j}","raw_label":"positive","raw_evidence_text":"e","raw_confidence":0.8,
                        "label":"positive","evidence_text":"e","confidence":0.8,"technical_status":"valid"}
                left.append(common.copy()); right.append(common.copy())
        right[0]["raw_label"]="uncertain"
        right[1]["label"]="uncertain"
        right[2]["technical_status"]="ambiguous_evidence_error"
        result=exp.compare_condition_rows(left,right)
        self.assertEqual(result["paired_cells"],60)
        self.assertEqual(len(result["raw_proposal_differences"]),1)
        self.assertEqual(len(result["accepted_row_differences"]),2)
        with self.assertRaises(ValueError): exp.compare_condition_rows(left[:-1],right[:-1])

    def test_parsers_and_runtime_are_source_identical(self):
        source=exp.staged_source_fingerprints()
        manifest=json.loads((REPO/"analysis/qwen_running_session_v1/configs/source_manifest.json").read_text())["source_sha256"]
        expected={
            "primary_v2": manifest["analysis/report_labeling_llm_v2/scripts/core.py"],
            "secondary_v1": manifest["analysis/report_labeling_llm_v1/scripts/benchmark_core.py"],
            "runtime_qwen": manifest["analysis/qwen_runtime_v1/scripts/runtime_qwen.py"],
        }
        self.assertEqual(source,expected)
        self.assertEqual(exp.sha_file(REPO/"analysis/qwen_running_session_v1/scripts/qwen_operator_run.py"),
            manifest["analysis/qwen_running_session_v1/scripts/qwen_operator_run.py"])

    def test_all_fixtures_are_synthetic_and_have_specifications(self):
        rows=json.loads((ROOT/"tests/fixtures/synthetic_cases.json").read_text())
        self.assertGreaterEqual(len(rows),8); self.assertLessEqual(len(rows),12)
        for row in rows:
            self.assertEqual(set(row),{"name","report","specification"})
            self.assertTrue(row["report"] and row["specification"])
        self.assertFalse(any("R1" in r["report"] or "StudyInstanceUID" in r["report"] for r in rows))


if __name__ == "__main__":
    unittest.main()
