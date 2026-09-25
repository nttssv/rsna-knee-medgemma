"""Prepare A/B prompt inputs offline; this script never loads Qwen weights or a GPU."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
REPO = ROOT.parents[1]
V2 = REPO / "analysis/report_labeling_llm_v2"
QWEN_LIVE = REPO / "analysis/qwen_execution_runtime_alt_v1/scripts/qwen_live.py"
CONFIG = json.loads((ROOT / "configs/experiment.json").read_text())
sys.path.insert(0, str(ROOT / "scripts"))
from experiment_core import build_prompt, sha_file, validate_inputs


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pinned module: {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def write_private(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())


def write_json(path: Path, value: object) -> None:
    write_private(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def run(prepared: Path, tokenizer_cache: Path, output: Path, completed_outputs: Path) -> dict:
    prepared, tokenizer_cache, output, completed_outputs = map(Path, (prepared, tokenizer_cache, output, completed_outputs))
    if output.exists():
        raise FileExistsError("Refuse to overwrite any prior preparation")
    plan = json.loads((prepared / "plan.json").read_text())
    if plan.get("files", {}).get("inputs.jsonl") != CONFIG["source_input_sha256"]:
        raise ValueError("Prepared input package differs from frozen input hash")
    if plan.get("origin", {}).get("split_sha256") != CONFIG["source_split_sha256"]:
        raise ValueError("Prepared split fingerprint differs")
    for filename, expected in plan["files"].items():
        if sha_file(prepared / filename) != expected:
            raise ValueError(f"Prepared artifact hash mismatch: {filename}")
    if plan.get("run_order") != ["control-1", "candidate-1", "candidate-2", "control-2"]:
        raise ValueError("Frozen source plan order differs")

    qwen_live = load("evidence_preparation_qwen_live", QWEN_LIVE)
    prepared_plan = qwen_live.read(prepared / "plan.json")
    rows = read_jsonl(prepared / "inputs.jsonl")
    with (prepared / "splits.csv").open(newline="") as f:
        split_rows = list(csv.DictReader(f))
    validate_inputs(rows, split_rows)
    actual_input_sha = sha_file(prepared / "inputs.jsonl")
    if actual_input_sha != CONFIG["source_input_sha256"]:
        raise ValueError("Frozen input package SHA mismatch")

    source_tokenized = json.loads((prepared / "tokenized.json").read_text())
    source_control_prompts = read_jsonl(prepared / "control_prompts.jsonl")
    if len(source_control_prompts) != 5:
        raise ValueError("Frozen control prompt package must contain five prompts")

    sys.path.insert(0, str(V2 / "scripts"))
    from v2_runtime import HFEncoder, messages_for
    encoder = HFEncoder("qwen", tokenizer_cache)
    def encode_cpu(prompt: str) -> dict:
        messages = messages_for("qwen", prompt)
        rendered = encoder.processor.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        ids = encoder.tokenizer(rendered, add_special_tokens=False,
            truncation=False)["input_ids"]
        return {"rendered_prompt": rendered, "input_ids": list(ids), "input_tokens": len(ids)}
    if encoder.metadata["model_revision"] != CONFIG["revision"]:
        raise ValueError("Pinned tokenizer revision mismatch")
    if (encoder.tokenizer.eos_token_id, encoder.tokenizer.pad_token_id) != (151645, 151643):
        raise ValueError("Pinned tokenizer stop/pad IDs mismatch")

    raw_runs = {}
    for block in ("control-1", "control-2"):
        raw_path = completed_outputs / block / "raw.jsonl"
        raw_runs[block] = read_jsonl(raw_path)
        if len(raw_runs[block]) != 5:
            raise ValueError(f"Completed reference block is incomplete: {block}")

    prompt_rows, tokenized, per_case = [], {"A": [], "B": []}, []
    for i, row in enumerate(rows):
        prompts = {variant: build_prompt(row["Report"], variant) for variant in ("A", "B")}
        if source_control_prompts[i] != {"case_index": i, "prompt": prompts["A"]}:
            raise ValueError(f"Control A prompt differs from frozen prepared input at case index {i}")
        prompt_rows.append({"case_index": i, "A": prompts["A"], "B": prompts["B"]})
        record = {"case_index": i, "report_sha256": row["report_sha256"], "language": row["language"], "tokens": {}}
        for variant in ("A", "B"):
            encoded = encode_cpu(prompts[variant])
            tokenized[variant].append({"rendered_prompt": encoded["rendered_prompt"],
                "input_ids": encoded["input_ids"], "input_tokens": encoded["input_tokens"]})
            record["tokens"][variant] = {"input_tokens": encoded["input_tokens"],
                "rendered_prompt_sha256": hashlib.sha256(encoded["rendered_prompt"].encode()).hexdigest()}
            if variant == "A":
                if tokenized[variant][-1] != source_tokenized["control"][i]:
                    raise ValueError(f"Control A token IDs are not exact prepared parity at case index {i}")
                for block in ("control-1", "control-2"):
                    old = raw_runs[block][i]
                    if (old.get("report_sha256") != row["report_sha256"] or
                            old.get("rendered_prompt") != encoded["rendered_prompt"] or
                            old.get("input_ids") != encoded["input_ids"]):
                        raise ValueError(f"Control A differs from completed run block {block} at case index {i}")
                if prompts[variant] not in tokenized[variant][-1]["rendered_prompt"]:
                    raise ValueError("A prompt is not present byte-for-byte in its rendered user input")
            if encoded["input_tokens"] > CONFIG["max_input_tokens"] or \
                    encoded["input_tokens"] + CONFIG["max_new_tokens"] > CONFIG["context_limit"]:
                raise ValueError(f"Input/context token cap exceeded for {variant}, case index {i}")
        per_case.append(record)

    output.mkdir(parents=True, mode=0o700)
    write_json(output / "prompt_inputs.json", prompt_rows)
    write_json(output / "tokenized.json", tokenized)
    write_json(output / "tokenizer_preflight.json", {
        "status": "PASS_LOCAL_ONLY", "model_id": CONFIG["model_id"], "revision": CONFIG["revision"],
        "tokenizer_metadata": encoder.metadata,
        "input_cap": CONFIG["max_input_tokens"], "output_cap": CONFIG["max_new_tokens"],
        "context_limit": CONFIG["context_limit"], "case_counts": per_case,
        "A_exact_prepared_token_parity": True, "A_exact_completed_run_parity": True,
        "B_within_input_and_context_caps": True, "model_weights_loaded": False,
        "gpu_accessed": False, "network_accessed": False})
    write_json(output / "input_fingerprints.json", {
        "prepared_plan_sha256": CONFIG["source_prepared_plan_sha256"],
        "prepared_plan_json_sha256": sha_file(prepared / "plan.json"),
        "inputs_sha256": actual_input_sha, "split_sha256": sha_file(prepared / "splits.csv"),
        "report_sha256": [r["report_sha256"] for r in rows],
        "study_ids_included": False, "organizer_labels_read": False})
    files = {p.name: sha_file(p) for p in sorted(output.iterdir())}
    write_json(output / "plan.json", {"status": "LOCAL_PROMPT_PLAN_NOT_RUN",
        "execution_enabled": False, "model_calls": 0, "source_prepared_plan_sha256": CONFIG["source_prepared_plan_sha256"],
        "inputs_sha256": actual_input_sha, "files_sha256": files,
        "prompt_sha256": {"A": sha_file(ROOT / "prompts/control_A.txt"),
                          "B": sha_file(ROOT / "prompts/candidate_B.txt"),
                          "definitions": sha_file(ROOT / "prompts/target_definitions.txt")},
        "run_order": CONFIG["run_order"], "maximum_generations": 20})
    return {"output": str(output), "plan_sha256": sha_file(output / "plan.json"), "tokenizer_preflight": "PASS_LOCAL_ONLY"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--tokenizer-cache", type=Path, required=True)
    parser.add_argument("--completed-outputs", type=Path, default=Path("state/runs/qwen-extraction-a6000-20260925/final/outputs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.prepared, args.tokenizer_cache, args.output, args.completed_outputs), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
