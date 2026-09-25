"""Supervised A/B Qwen report extraction. Default action is an offline dry run."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
REPO = ROOT.parents[1]
PLAN = ROOT / "configs/execution_plan.json"
PREPARED_SHA = "a26965fa84d7d2db37d8d3812f7b4061119df4bf77690127bb27d78e0f8eda82"
A_SHA = "86b3fdf3adffff5e74b3b94192580c5d771711d441d682ecba88cc64bb344a68"
B_SHA = "f1b39d642bd4d7948be7d9460897641400968fe87657529e725507176a044ff1"
DEFINITIONS_SHA = "dab72cb29da5105bc6f02336b50c3488028d92b0cf1b1bfe78f1038f6fd1264b"
SOURCE_SHA = "c4586bb0c65163244dac80344ea96f691493a1b89a5911eefcba54f2384c74fc"
ORDER = ("A1", "B1", "B2", "A2")
REQUIRED_SOURCES = {
    "analysis/qwen_evidence_selection_v1/scripts/run_ab.py",
    "analysis/qwen_evidence_selection_v1/scripts/stop_at.py",
    "analysis/qwen_evidence_selection_v1/scripts/experiment_core.py",
    "analysis/qwen_evidence_selection_v1/configs/experiment.json",
    "analysis/qwen_evidence_selection_v1/prompts/control_A.txt",
    "analysis/qwen_evidence_selection_v1/prompts/candidate_B.txt",
    "analysis/qwen_evidence_selection_v1/prompts/target_definitions.txt",
    "analysis/qwen_execution_runtime_alt_v1/scripts/qwen_live.py",
    "analysis/qwen_runtime_v1/scripts/runtime_qwen.py",
    "analysis/report_labeling_llm_v2/scripts/v2_runtime.py",
    "analysis/report_labeling_llm_v2/scripts/core.py",
    "analysis/report_labeling_llm_v1/scripts/benchmark_core.py",
    "analysis/report_labeling_llm_v2/scripts/run_smoke.py",
    "analysis/report_labeling_llm_v2/scripts/load_preflight.py",
}
ALLOWLIST = {"NVIDIA A40", "NVIDIA RTX A6000", "NVIDIA L40",
             "NVIDIA RTX 6000 Ada Generation", "NVIDIA L40S"}
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(REPO / "analysis/report_labeling_llm_v2/scripts"))
from experiment_core import build_prompt, validate_inputs
from run_smoke import supervise


class GateError(RuntimeError):
    pass


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read(path: Path):
    return json.loads(Path(path).read_text())


def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def live_runtime():
    return load_module("qwen_evidence_existing_live_runtime",
                       REPO / "analysis/qwen_execution_runtime_alt_v1/scripts/qwen_live.py")


def private(path: Path) -> Path:
    path = Path(path).resolve()
    if path.is_relative_to(REPO) and not path.is_relative_to(REPO / "state"):
        raise GateError("Private artifacts must be in ignored state/ or an external private directory")
    return path


def private_json(path: Path):
    path = private(path)
    mode = path.lstat()
    if not stat.S_ISREG(mode.st_mode) or stat.S_IMODE(mode.st_mode) != 0o600 or mode.st_uid != os.getuid():
        raise GateError("Authorization, observation and stop receipts must be owner-only 0600 files")
    return read(path)


def write_new(path: Path, value) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def append(stream, value) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def aware(value: str) -> datetime:
    if not isinstance(value, str):
        raise GateError("Timezone-aware timestamp required")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise GateError("Timezone-aware timestamp required")
    return parsed.astimezone(timezone.utc)


def validate_plan(expected_sha: str) -> dict:
    if sha(PLAN) != expected_sha:
        raise GateError("New execution-plan SHA mismatch")
    plan = read(PLAN)
    if (plan.get("schema_version") != 1 or plan.get("experiment") != "qwen_evidence_selection_v1"
        or plan.get("default_action") != "dry-run" or plan.get("prepared_plan_sha256") != PREPARED_SHA
        or plan.get("source_prepared_plan_sha256") != SOURCE_SHA
        or plan.get("prompt_sha256") != {"A": A_SHA, "B": B_SHA, "definitions": DEFINITIONS_SHA}
        or set(plan.get("source_sha256", {})) != REQUIRED_SOURCES
        or plan.get("run_order") != list(ORDER) or plan.get("maximum_generations") != 20
        or plan.get("hard_inference_seconds") != 1800
        or plan.get("copy_stop_reserve_seconds") != 900
        or plan.get("model_id") != "Qwen/Qwen3-14B"
        or plan.get("model_revision") != "40c069824f4251a91eefaf281ebe4c544efd3e18"
        or plan.get("precision") != "bfloat16" or plan.get("attention") != "sdpa"
        or plan.get("batch_size") != 1 or plan.get("seed") != 20260914
        or plan.get("enable_thinking") is not False or plan.get("do_sample") is not False
        or plan.get("max_input_tokens") != 8192 or plan.get("max_new_tokens") != 2048
        or plan.get("context_limit") != 40960 or plan.get("stop_token_ids") != [151645, 151643]
        or plan.get("reports_per_block") != 5 or plan.get("max_provider_seconds") != 10800
        or plan.get("max_session_usd") != 3.0):
        raise GateError("Execution-plan scope differs from the fixed A/B experiment")
    for relative, expected in plan["source_sha256"].items():
        if sha(REPO / relative) != expected:
            raise GateError(f"Execution source changed: {relative}")
    cfg = read(ROOT / "configs/experiment.json")
    if (cfg["run_order"] != list(ORDER) or cfg["maximum_generations"] != 20
        or cfg["candidate_B_sha256"] != B_SHA or cfg["execution_enabled"] is not False
        or cfg["revision"] != "40c069824f4251a91eefaf281ebe4c544efd3e18"):
        raise GateError("Scientific A/B configuration changed")
    return plan


def load_package(prepared: Path, source: Path, plan: dict):
    prepared, source = private(prepared), private(source)
    if sha(prepared / "plan.json") != PREPARED_SHA or sha(source / "plan.json") != SOURCE_SHA:
        raise GateError("Prepared A/B or original five-report plan SHA changed")
    package, source_plan = read(prepared / "plan.json"), read(source / "plan.json")
    if (package.get("execution_enabled") is not False or package.get("model_calls") != 0
        or package.get("run_order") != list(ORDER) or package.get("maximum_generations") != 20
        or package.get("inputs_sha256") != plan["inputs_sha256"]
        or package.get("source_prepared_plan_sha256") != SOURCE_SHA
        or package.get("prompt_sha256") != plan["prompt_sha256"]):
        raise GateError("A/B prepared-package contract changed")
    for name, expected in package["files_sha256"].items():
        if name not in {"prompt_inputs.json", "tokenized.json", "tokenizer_preflight.json", "input_fingerprints.json"} \
                or sha(prepared / name) != expected:
            raise GateError(f"A/B prepared file changed: {name}")
    for name, expected in source_plan["files"].items():
        if sha(source / name) != expected:
            raise GateError(f"Original five-report prepared file changed: {name}")
    if sha(source / "inputs.jsonl") != plan["inputs_sha256"] or sha(source / "splits.csv") != plan["split_sha256"]:
        raise GateError("Original input or 40/18 split changed")
    rows = read_lines(source / "inputs.jsonl")
    with (source / "splits.csv").open(newline="") as stream:
        validate_inputs(rows, list(csv.DictReader(stream)))
    prompts, tokens = read(prepared / "prompt_inputs.json"), read(prepared / "tokenized.json")
    if len(prompts) != 5 or set(tokens) != {"A", "B"} or any(len(tokens[k]) != 5 for k in tokens):
        raise GateError("Exactly five A and five B prompts/token entries required")
    for index, row in enumerate(rows):
        if prompts[index] != {"case_index": index,
                              "A": build_prompt(row["Report"], "A"),
                              "B": build_prompt(row["Report"], "B")}:
            raise GateError("Prompt package does not match the fixed A/B prompt and report")
        for arm in ("A", "B"):
            entry = tokens[arm][index]
            if (set(entry) != {"rendered_prompt", "input_ids", "input_tokens"}
                or type(entry["input_tokens"]) is not int or entry["input_tokens"] != len(entry["input_ids"])
                or entry["input_tokens"] > 8192 or entry["input_tokens"] + 2048 > 40960):
                raise GateError("Input-token package or cap changed")
    return rows, prompts, tokens


def encode_checked(encoder, rows, prompts, tokens, *, synthetic: bool):
    from v2_runtime import messages_for
    result = {"A": [], "B": []}
    for arm in ("A", "B"):
        for index, row in enumerate(rows):
            prompt = prompts[index][arm]
            if synthetic:
                rendered = encoder.processor.apply_chat_template(messages_for("qwen", prompt),
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
                ids = encoder.tokenizer(rendered, add_special_tokens=False, truncation=False)["input_ids"]
                encoded = SimpleNamespace(rendered_prompt=rendered, input_ids=list(ids), input_tokens=len(ids))
            else:
                encoded = encoder.encode(prompt)  # HFEncoder creates the actual tokenizer tensors.
                if not hasattr(encoded, "inputs"):
                    raise GateError("Real encoder did not produce backend tokenizer tensors")
            expected = tokens[arm][index]
            if (encoded.rendered_prompt != expected["rendered_prompt"]
                or encoded.input_ids != expected["input_ids"]
                or encoded.input_tokens != expected["input_tokens"]
                or prompt not in encoded.rendered_prompt):
                raise GateError(f"Rendered input/token mismatch for {arm} report {index}")
            result[arm].append(encoded)
    return result


def validate_session(session: dict, observation: dict, stop: dict, plan_sha: str,
                     session_file_sha256: str, *, now: datetime | None = None) -> float:
    required = {"schema_version", "approved_by_user", "execution_plan_sha256", "prepared_plan_sha256",
        "prompt_sha256", "pod_id", "region", "gpu_name", "gpu_count", "cloud_type", "container_disk_gb",
        "persistent_volume_gb", "network_volume_id", "compute_usd_per_hour", "storage_usd_per_hour",
        "maximum_usd", "t0", "hard_deadline", "shutdown_at"}
    if set(session) != required or session["schema_version"] != 1 or session["approved_by_user"] is not True:
        raise GateError("A fresh, exact private user-approval record is required")
    if (session["execution_plan_sha256"] != plan_sha or session["prepared_plan_sha256"] != PREPARED_SHA
        or session["prompt_sha256"] != read(PLAN)["prompt_sha256"]):
        raise GateError("Approval does not bind this execution and prompt plan")
    if not isinstance(session["pod_id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", session["pod_id"]):
        raise GateError("Exact approved pod ID required")
    expected = {"region": session["region"], "gpu_name": session["gpu_name"],
        "gpu_count": 1, "cloud_type": "SECURE", "container_disk_gb": 80,
        "persistent_volume_gb": 0, "network_volume_id": None}
    if (session["gpu_name"] not in ALLOWLIST or not isinstance(session["region"], str)
        or not session["region"] or any(session.get(k) != v or type(session.get(k)) is not type(v)
                                   for k, v in expected.items())):
        raise GateError("Approved resource differs from the reviewed 48 GB Secure Cloud boundary")
    for key in ("compute_usd_per_hour", "storage_usd_per_hour", "maximum_usd"):
        value = session[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise GateError("Approved rates and budget must be finite and nonnegative")
    if not 0 < session["maximum_usd"] <= 3 or session["compute_usd_per_hour"] <= 0:
        raise GateError("Approved budget or compute rate is invalid")
    t0, deadline, shutdown = map(aware, (session["t0"], session["hard_deadline"], session["shutdown_at"]))
    now = now or datetime.now(timezone.utc)
    seconds = (deadline - t0).total_seconds()
    if (not 0 < seconds <= 10800 or not t0 <= now < shutdown <= deadline
        or (deadline - shutdown).total_seconds() < 300
        or (session["compute_usd_per_hour"] + session["storage_usd_per_hour"]) * seconds / 3600 > session["maximum_usd"]):
        raise GateError("Immutable whole-session deadline/budget or stop reserve invalid")
    observed_fields = {"pod_id", "provider_state", "observed_at", "region", "gpu_name", "gpu_count",
        "cloud_type", "container_disk_gb", "persistent_volume_gb", "network_volume_id",
        "actual_compute_usd_per_hour", "actual_storage_usd_per_hour", "current_total_usd_per_hour"}
    if set(observation) != observed_fields or observation["pod_id"] != session["pod_id"] \
            or observation["provider_state"] != "RUNNING":
        raise GateError("Fresh exact-pod RUNNING observation required")
    for key in ("compute_usd_per_hour", "storage_usd_per_hour"):
        value = observation["actual_" + key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise GateError("Observed provider rate is unavailable or invalid")
    if any(observation[k] != session[k] or type(observation[k]) is not type(session[k])
           for k in expected) or any(observation["actual_" + k] > session[k]
                                    for k in ("compute_usd_per_hour", "storage_usd_per_hour")):
        raise GateError("Observed provider resource or rate differs from approval")
    total = observation["current_total_usd_per_hour"]
    if type(total) not in (int, float) or not math.isfinite(total) or total < 0 \
            or abs(total - observation["actual_compute_usd_per_hour"]
                   - observation["actual_storage_usd_per_hour"]) > 0.001 \
            or total > session["compute_usd_per_hour"] + session["storage_usd_per_hour"] + 0.001:
        raise GateError("Observed total hourly billing exceeds the approval")
    observed_at = aware(observation["observed_at"])
    if not t0 <= observed_at <= now or (now - observed_at).total_seconds() > 600:
        raise GateError("Provider observation is stale or outside the approved session")
    if (set(stop) != {"pod_id", "execution_plan_sha256", "session_sha256", "shutdown_at", "checked_at",
                      "watchdog_pid", "watchdog_argv"}
        or stop["pod_id"] != session["pod_id"] or stop["execution_plan_sha256"] != plan_sha
        or stop["session_sha256"] != session_file_sha256 or stop["shutdown_at"] != session["shutdown_at"]):
        raise GateError("Pod-local shutdown process receipt is missing or mismatched")
    if (now - aware(stop["checked_at"])).total_seconds() > 600 or aware(stop["checked_at"]) > now:
        raise GateError("Pod-local shutdown receipt is stale")
    verify_watchdog(stop)
    room = min(1799.0, (deadline - now).total_seconds() - 900.0, (shutdown - now).total_seconds() - 1.0)
    if room < 1799:
        raise GateError("Insufficient time for the 1800-second inference cap and copy/stop reserve")
    return room


def verify_watchdog(stop: dict) -> None:
    pid, argv = stop.get("watchdog_pid"), stop.get("watchdog_argv")
    if type(pid) is not int or pid <= 1 or not isinstance(argv, list) or len(argv) < 5 \
            or not all(isinstance(x, str) and x for x in argv) or stop["pod_id"] not in argv \
            or str(ROOT / "scripts/stop_at.py") not in argv \
            or stop["execution_plan_sha256"] not in argv:
        raise GateError("Pod-local exact-pod shutdown command is invalid")
    proc = Path(f"/proc/{pid}/cmdline")
    if not proc.is_file() or proc.read_bytes().rstrip(b"\0").split(b"\0") != [x.encode() for x in argv]:
        raise GateError("Pod-local shutdown process is not live at its approved command")


def check_live_step(session: dict, stop: dict) -> None:
    now = datetime.now(timezone.utc)
    if now >= aware(session["hard_deadline"]) or now >= aware(session["shutdown_at"]) \
            or (aware(session["hard_deadline"]) - now).total_seconds() <= 900:
        raise TimeoutError("Inference reached its fixed copy/stop reserve")
    verify_watchdog(stop)


def verify_hardware(gpu_name: str, runtime) -> dict:
    import torch
    from v2_runtime import check_versions
    versions = check_versions()
    normalized = dict(versions)
    if normalized.get("torch") == "2.8.0+cu128":
        normalized["torch"] = "2.8.0"
    if normalized != runtime.runtime_config()["required_versions"]:
        raise GateError("Pinned package versions differ")
    def query(field):
        output = subprocess.run(["nvidia-smi", "--query-gpu=" + field, "--format=csv,noheader"],
                                check=True, capture_output=True, text=True, timeout=15)
        return [line.strip() for line in output.stdout.splitlines() if line.strip()]
    names, drivers = query("name"), query("driver_version")
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader"], check=True, capture_output=True, text=True, timeout=15)
    if processes.stdout.strip() or len(drivers) != 1:
        raise GateError("Another GPU workload exists or driver provenance is unavailable")
    runtime.verify_reported_hardware(names, torch.cuda.device_count(), torch.version.cuda,
        torch.__version__, torch.cuda.is_available(),
        torch.cuda.is_bf16_supported(including_emulation=False),
        torch.cuda.mem_get_info(0)[0] / 2**30, gpu_name)
    return {"gpu_names": names, "driver_version": drivers[0], "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__, "package_versions": versions,
            "free_gpu_gib": torch.cuda.mem_get_info(0)[0] / 2**30}


class FakeBackend:
    def __init__(self, encoder, behavior: str):
        self.encoder, self.behavior = encoder, behavior
        self.eos, self.calls = [151645, 151643], 0
        self.metadata = {"synthetic": True, "model_weights_loaded": False}

    def peak_gib(self):
        return 0.0

    def generate(self, encoded):
        self.calls += 1
        if self.behavior == "hang":
            time.sleep(5)
        if self.behavior == "raise":
            raise RuntimeError("synthetic_generation_exception")
        from core import LABELS
        values = {key: {"label": "not_mentioned", "evidence_text": "", "confidence": 0} for key in LABELS}
        raw = json.dumps(values, ensure_ascii=False)
        if self.behavior == "schema_error":
            raw = "{}"
        ids = self.encoder.tokenizer(raw, add_special_tokens=False)["input_ids"] + [151645]
        if self.behavior == "incomplete":
            ids[-1] = self.encoder.tokenizer("?", add_special_tokens=False)["input_ids"][0]
        body = ids[:-1] if ids[-1] in self.eos else ids
        return {"raw_output": self.encoder.decode(body),
                "decoded_with_special_tokens": self.encoder.decode(ids),
                "output_token_ids": ids, "input_tokens": encoded.input_tokens,
                "output_tokens": len(ids), "runtime_seconds": 0.01,
                "generation_status": "completed" if self.behavior != "incomplete" else "max_new_tokens"}


def run_worker(args) -> bool:
    plan = validate_plan(args.plan_sha256)
    rows, prompts, tokens = load_package(args.prepared, args.source_prepared, plan)
    synthetic = args.rehearsal
    session = observation = stop = None
    timeout = None
    if not synthetic:
        session, observation, stop = (private_json(args.session), private_json(args.observation),
                                      private_json(args.shutdown_receipt))
        timeout = validate_session(session, observation, stop, args.plan_sha256, sha(args.session))
    from v2_runtime import HFEncoder
    runtime = live_runtime()
    if synthetic:
        encoder = HFEncoder("qwen", args.cache)
    else:
        hardware = verify_hardware(session["gpu_name"], runtime)
        from load_preflight import audit_cache
        audit = audit_cache(args.cache, "qwen")
        write_new(args.output / "cache_audit.json", audit)
        if not audit["complete"] or audit["revision"] != plan["model_revision"]:
            raise GateError("Pinned Qwen cache is incomplete")
        encoder = runtime.create_encoder(args.cache)
    arms = encode_checked(encoder, rows, prompts, tokens, synthetic=synthetic)
    write_new(args.output / "input_preflight.json", {"synthetic": synthetic,
        "exact_prompt_and_token_parity": True, "prepared_plan_sha256": PREPARED_SHA,
        "prompt_sha256": plan["prompt_sha256"],
        "input_tokens": {arm: [e.input_tokens for e in arm_rows] for arm, arm_rows in arms.items()}})
    if synthetic:
        backend = FakeBackend(encoder, args.fake_behavior)
    else:
        check_live_step(session, stop)
        backend = runtime.create_backend(encoder, args.cache, session["gpu_name"])
        write_new(args.output / "model_preflight.json", {"hardware": hardware,
            "backend": backend.metadata, "eos_ids": backend.eos, "revision": plan["model_revision"]})
    result = {"status": "failed", "synthetic": synthetic, "planned": 20, "attempted": 0,
              "completed": 0, "failed": 0, "unrun": 20, "blocks_completed": [],
              "execution_plan_sha256": args.plan_sha256, "started_at": datetime.now(timezone.utc).isoformat()}
    started = time.monotonic()
    try:
        for block in ORDER:
            arm = block[0]
            directory = args.output / block
            directory.mkdir(mode=0o700)
            with (directory / "attempts.jsonl").open("x") as attempts, \
                 (directory / "raw.jsonl").open("x") as raw, \
                 (directory / "parsed.jsonl").open("x") as parsed, \
                 (directory / "technical_failures.jsonl").open("x") as failures:
                for stream in (attempts, raw, parsed, failures):
                    os.chmod(stream.name, 0o600)
                for index, (row, encoded) in enumerate(zip(rows, arms[arm])):
                    if result["attempted"] >= 20:
                        raise GateError("Twenty-generation maximum reached")
                    if not synthetic:
                        check_live_step(session, stop)
                    number = result["attempted"] + 1
                    append(attempts, {"synthetic": synthetic, "attempt": number, "block": block,
                        "arm": arm, "case_index": index, "report_sha256": row["report_sha256"],
                        "prompt_sha256": plan["prompt_sha256"][arm],
                        "started_at": datetime.now(timezone.utc).isoformat()})
                    result["attempted"] = number
                    try:
                        generation = backend.generate(encoded)
                    except BaseException as exc:
                        append(failures, {"synthetic": synthetic, "attempt": number,
                            "kind": "generation_exception", "error_type": type(exc).__name__})
                        result["failed"] += 1
                        raise
                    record = {"synthetic": synthetic, "attempt": number, "block": block,
                        "arm": arm, "case_index": index, "StudyInstanceUID": row["StudyInstanceUID"],
                        "report_sha256": row["report_sha256"], "prompt_sha256": plan["prompt_sha256"][arm],
                        "rendered_prompt": encoded.rendered_prompt, "input_ids": encoded.input_ids,
                        "input_tokens": encoded.input_tokens, "generation": generation,
                        "gpu_peak_allocated_gib": backend.peak_gib()}
                    append(raw, record)  # Raw and token IDs are durable before decode or parsing.
                    result["completed"] += generation.get("generation_status") == "completed"
                    try:
                        primary, secondary = runtime.validate_generation(generation, backend, encoded, row["Report"])
                    except BaseException as exc:
                        append(failures, {"synthetic": synthetic, "attempt": number,
                            "kind": "integrity_or_incomplete_generation", "error_type": type(exc).__name__})
                        result["failed"] += 1
                        raise
                    append(parsed, {"synthetic": synthetic, "attempt": number, "arm": arm,
                        "prompt_sha256": plan["prompt_sha256"][arm], "primary_v2": primary,
                        "secondary_v1": secondary, "raw_sha256": hashlib.sha256(
                            json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()})
                    errors = [r["status"] for r in primary["rows"] if r["status"] != "valid"]
                    if errors:
                        append(failures, {"synthetic": synthetic, "attempt": number,
                                          "kind": "parser_technical_failure", "condition_statuses": errors})
            write_new(args.output / (block + ".complete.json"), {"synthetic": synthetic,
                "block": block, "completed_generations": 5,
                "finished_at": datetime.now(timezone.utc).isoformat()})
            result["blocks_completed"].append(block)
    except BaseException as exc:
        result["failure_type"] = type(exc).__name__
    finally:
        result.update(status="completed" if result["attempted"] == 20 and result["failed"] == 0 else "failed",
            unrun=20-result["attempted"], elapsed_seconds=time.monotonic()-started,
            peak_gpu_allocated_gib=backend.peak_gib(),
            finished_at=datetime.now(timezone.utc).isoformat(),
            artifacts_sha256={str(p.relative_to(args.output)): sha(p) for p in sorted(args.output.rglob("*"))
                              if p.is_file() and p.name != "session_result.json"})
        write_new(args.output / "session_result.json", result)
    return result["status"] == "completed"


def verify_child(args) -> None:
    if args.worker_fd is None or args.worker_fd < 3 or not stat.S_ISFIFO(os.fstat(args.worker_fd).st_mode):
        raise GateError("Worker requires the active supervisor pipe")
    with os.fdopen(args.worker_fd) as stream:
        message = json.loads(stream.read(4096))
    start_path, dispatch_path = args.output / "supervisor_start.json", args.output / "supervisor_dispatch.json"
    start, dispatch = read(start_path), read(dispatch_path)
    if (message.get("parent_pid") != os.getppid() or start.get("parent_pid") != os.getppid()
        or message.get("plan_sha256") != args.plan_sha256 or start.get("plan_sha256") != args.plan_sha256
        or message.get("nonce") is None or hashlib.sha256(message["nonce"].encode()).hexdigest() != dispatch.get("nonce_sha256")
        or dispatch.get("start_sha256") != sha(start_path)
        or start.get("prepared_plan_sha256") != PREPARED_SHA
        or (args.output / "session_result.json").exists()):
        raise GateError("Worker lacks active parent attestation")
    if not args.rehearsal and (start.get("session_sha256") != sha(args.session)
        or start.get("observation_sha256") != sha(args.observation)
        or start.get("shutdown_receipt_sha256") != sha(args.shutdown_receipt)):
        raise GateError("Private authorization receipts changed after supervisor launch")


def launch(args, timeout: float):
    if args.output.exists():
        raise FileExistsError("Never overwrite or resume a previous output directory")
    args.output.mkdir(mode=0o700, parents=True)
    write_new(args.output / "supervisor_start.json", {"synthetic": args.rehearsal,
        "parent_pid": os.getpid(), "plan_sha256": args.plan_sha256,
        "prepared_plan_sha256": PREPARED_SHA,
        "session_sha256": None if args.rehearsal else sha(args.session),
        "observation_sha256": None if args.rehearsal else sha(args.observation),
        "shutdown_receipt_sha256": None if args.rehearsal else sha(args.shutdown_receipt),
        "started_at": datetime.now(timezone.utc).isoformat()})
    nonce = secrets.token_hex(32)
    write_new(args.output / "supervisor_dispatch.json", {"start_sha256": sha(args.output / "supervisor_start.json"),
        "nonce_sha256": hashlib.sha256(nonce.encode()).hexdigest()})
    pipe_r, pipe_w = os.pipe()
    command = [sys.executable, str(HERE), "--worker", "--prepared", str(args.prepared),
        "--source-prepared", str(args.source_prepared), "--cache", str(args.cache),
        "--output", str(args.output), "--plan-sha256", args.plan_sha256, "--worker-fd", str(pipe_r)]
    if args.rehearsal:
        command.extend(["--rehearsal", "--fake-behavior", args.fake_behavior])
    else:
        command.extend(["--run", "--session", str(args.session), "--observation", str(args.observation),
                        "--shutdown-receipt", str(args.shutdown_receipt)])
    with os.fdopen(pipe_w, "w") as stream:
        json.dump({"parent_pid": os.getpid(), "plan_sha256": args.plan_sha256, "nonce": nonce}, stream)
    try:
        okay, status = supervise(command, timeout, pass_fds=(pipe_r,))
    finally:
        os.close(pipe_r)
    write_new(args.output / "supervisor_result.json", {"synthetic": args.rehearsal,
        "status": status, "worker_exit_success": okay, "hard_timeout_seconds": timeout,
        "partial_artifacts_sha256": {str(p.relative_to(args.output)): sha(p) for p in sorted(args.output.rglob("*"))
                                     if p.is_file() and p.name != "supervisor_result.json"}})
    return okay, status


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("prepared", "source-prepared", "cache", "output", "session", "observation", "shutdown-receipt"):
        parser.add_argument("--" + flag, type=Path)
    parser.add_argument("--plan-sha256", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--rehearsal", action="store_true", help="Supervised 20-call SYNTHETIC CPU run")
    group.add_argument("--run", action="store_true", help="Approved real Qwen generation")
    parser.add_argument("--fake-behavior", choices=("normal", "hang", "raise", "incomplete", "schema_error"),
                        default="normal", help=argparse.SUPPRESS)
    parser.add_argument("--rehearsal-timeout", type=float, default=60.0, help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    required = (args.prepared, args.source_prepared, args.cache)
    if any(x is None for x in required):
        parser.error("--prepared, --source-prepared and --cache are required")
    args.prepared, args.source_prepared, args.cache = map(Path, required)
    validate_plan(args.plan_sha256)
    if args.worker:
        if args.output is None or not (args.run or args.rehearsal):
            raise GateError("Worker requires the complete parent dispatch")
        args.output = private(args.output)
        verify_child(args)
        return 0 if run_worker(args) else 1
    plan = read(PLAN)
    rows, prompts, tokens = load_package(args.prepared, args.source_prepared, plan)
    if not args.run and not args.rehearsal:
        from v2_runtime import HFEncoder
        encoder = HFEncoder("qwen", args.cache)
        encode_checked(encoder, rows, prompts, tokens, synthetic=True)
        print(json.dumps({"status": "DRY_RUN_PASS", "execution_plan_sha256": args.plan_sha256,
            "prepared_plan_sha256": PREPARED_SHA, "exact_prompt_and_token_parity": True,
            "generations_executed": 0}, indent=2))
        return 0
    if args.output is None:
        parser.error("--output is required for rehearsal or real execution")
    args.output = private(args.output)
    if args.rehearsal:
        if not 0 < args.rehearsal_timeout <= 60:
            raise GateError("Synthetic supervisor timeout must be 0-60 seconds")
        timeout = args.rehearsal_timeout
    else:
        if args.fake_behavior != "normal" or args.rehearsal_timeout != 60:
            raise GateError("Fake controls cannot be used for real execution")
        if any(x is None for x in (args.session, args.observation, args.shutdown_receipt)):
            raise GateError("Real run needs a fresh user approval, provider observation and live shutdown receipt")
        args.session, args.observation, args.shutdown_receipt = map(private,
            (args.session, args.observation, args.shutdown_receipt))
        timeout = validate_session(private_json(args.session), private_json(args.observation),
                                   private_json(args.shutdown_receipt), args.plan_sha256, sha(args.session))
    okay, status = launch(args, timeout)
    print(json.dumps({"status": status, "synthetic": args.rehearsal,
        "worker_exit_success": okay, "output": str(args.output)}, indent=2))
    return 0 if okay else 1


if __name__ == "__main__":
    raise SystemExit(main())
