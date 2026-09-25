"""Local-only prompt and paired-output helpers for the Qwen evidence experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
PAIRED_FIELDS = ("label", "evidence_text", "confidence", "technical_status")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(Path(path).read_bytes())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def build_prompt(report: str, variant: str) -> str:
    if not isinstance(report, str) or not report:
        raise ValueError("Nonempty original-language report required")
    if variant not in {"A", "B"}:
        raise ValueError("Variant must be A or B")
    base = (ROOT / "prompts/control_A.txt").read_text()
    definitions = (ROOT / "prompts/target_definitions.txt").read_text()
    if variant == "B":
        base = (ROOT / "prompts/candidate_B.txt").read_text()
    return (base + "\nTARGET DEFINITIONS:\n" + definitions +
            "\nREPORT_JSON_STRING:\n" + json.dumps(report, ensure_ascii=False))


def validate_inputs(rows: list[dict], split_rows: list[dict]) -> None:
    if len(rows) != 5 or len({r.get("StudyInstanceUID") for r in rows}) != 5:
        raise ValueError("Exactly five unique prepared reports required")
    if len(split_rows) != 58 or sum(r.get("split") == "development" for r in split_rows) != 40 \
            or sum(r.get("split") == "validation" for r in split_rows) != 18:
        raise ValueError("Frozen 40/18 split mismatch")
    dev = {r["StudyInstanceUID"] for r in split_rows if r.get("split") == "development"}
    for row in rows:
        if set(row) != {"StudyInstanceUID", "Report", "split", "language", "report_sha256"}:
            raise ValueError("Prepared row contains missing, unexpected or label-bearing fields")
        if row["split"] != "development" or row["StudyInstanceUID"] not in dev:
            raise ValueError("Only the five prepared development reports are permitted")
        if not isinstance(row["Report"], str) or not row["Report"]:
            raise ValueError("Original-language report is empty or invalid")
        if sha_bytes(row["Report"].encode("utf-8")) != row["report_sha256"]:
            raise ValueError("Report source hash mismatch")


def compare_condition_rows(left: list[dict], right: list[dict]) -> dict:
    """Keep model-proposed and accepted-parser differences distinct."""
    key = lambda row: (row["case_index"], row["condition"])
    a, b = {key(x): x for x in left}, {key(x): x for x in right}
    if len(a) != 60 or len(b) != 60 or a.keys() != b.keys():
        raise ValueError("Each block must contain the same 60 report-condition cells")
    raw_fields = ("raw_label", "raw_evidence_text", "raw_confidence")
    parsed_fields = PAIRED_FIELDS
    raw_changes, accepted_changes = [], []
    for k in sorted(a):
        changed_raw = [f for f in raw_fields if a[k].get(f) != b[k].get(f)]
        changed_accepted = [f for f in parsed_fields if a[k].get(f) != b[k].get(f)]
        if changed_raw:
            raw_changes.append({"case_index": k[0], "condition": k[1], "fields": changed_raw})
        if changed_accepted:
            accepted_changes.append({"case_index": k[0], "condition": k[1], "fields": changed_accepted})
    return {"paired_cells": 60, "raw_proposal_differences": raw_changes,
            "accepted_row_differences": accepted_changes}


def verify_source_span(report: str, evidence: str, start: int, end: int) -> bool:
    return isinstance(evidence, str) and 0 <= start < end <= len(report) \
        and report[start:end] == evidence


def staged_source_fingerprints() -> dict[str, str]:
    paths = {
        "primary_v2": REPO / "analysis/report_labeling_llm_v2/scripts/core.py",
        "secondary_v1": REPO / "analysis/report_labeling_llm_v1/scripts/benchmark_core.py",
        "runtime_qwen": REPO / "analysis/qwen_runtime_v1/scripts/runtime_qwen.py",
    }
    return {name: sha_file(path) for name, path in paths.items()}
