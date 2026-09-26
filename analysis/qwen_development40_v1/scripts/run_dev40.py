"""One-pass Qwen extraction for the frozen 40-study development split.

The default action is a zero-generation offline dry run. Real generation requires
an explicit --run plus fresh private authorization, provider observation and
live shutdown receipt. Organizer labels are never loaded by this program.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import secrets
import stat
import sys
import time

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
REPO = ROOT.parents[1]
PLAN = ROOT / "configs/execution_plan.json"
PREPARED_PLAN_SHA = "b2c6f7e4d220ab83f5556170334fa441dac28c495e3ab9e9b8fefc8e956ca023"
CONTROL_SHA = "86b3fdf3adffff5e74b3b94192580c5d771711d441d682ecba88cc64bb344a68"
DEFINITIONS_SHA = "dab72cb29da5105bc6f02336b50c3488028d92b0cf1b1bfe78f1038f6fd1264b"
REPORTS = 40
MAX_GENERATIONS = 40
ALLOWLIST = {"NVIDIA A40", "NVIDIA RTX A6000", "NVIDIA L40",
             "NVIDIA RTX 6000 Ada Generation", "NVIDIA L40S"}
REQUIRED_SOURCES = {
    "analysis/qwen_development40_v1/scripts/run_dev40.py",
    "analysis/qwen_evidence_selection_v1/scripts/run_ab.py",
    "analysis/qwen_evidence_selection_v1/scripts/stop_at.py",
    "analysis/qwen_evidence_selection_v1/prompts/control_A.txt",
    "analysis/qwen_evidence_selection_v1/prompts/target_definitions.txt",
    "analysis/qwen_execution_runtime_alt_v1/scripts/qwen_live.py",
    "analysis/qwen_execution_runtime_alt_v1/configs/runtime.json",
    "analysis/qwen_runtime_v1/scripts/runtime_qwen.py",
    "analysis/qwen_report_extraction_v1/configs/experiment.json",
    "analysis/report_labeling_llm_v2/scripts/v2_runtime.py",
    "analysis/report_labeling_llm_v2/scripts/core.py",
    "analysis/report_labeling_llm_v2/scripts/run_smoke.py",
    "analysis/report_labeling_llm_v2/scripts/load_preflight.py",
    "analysis/report_labeling_llm_v2/configs/qwen.json",
    "analysis/report_labeling_llm_v2/configs/runtime.json",
    "analysis/report_labeling_llm_v1/scripts/benchmark_core.py",
}


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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ab_runtime():
    return load_module("qwen_dev40_ab_helpers", REPO / "analysis/qwen_evidence_selection_v1/scripts/run_ab.py")


def live_runtime():
    return load_module("qwen_dev40_live_runtime",
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
        raise GateError("Private authorization receipts must be owner-only 0600 files")
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
        raise GateError("Execution-plan SHA mismatch")
    plan = read(PLAN)
    expected = {
        "schema_version": 1,
        "experiment": "qwen_development40_v1",
        "default_action": "dry-run",
        "execution_enabled": False,
        "prepared_plan_sha256": PREPARED_PLAN_SHA,
        "prompt_sha256": CONTROL_SHA,
        "definitions_sha256": DEFINITIONS_SHA,
        "reports": REPORTS,
        "passes": 1,
        "maximum_generations": MAX_GENERATIONS,
        "model_id": "Qwen/Qwen3-14B",
        "model_revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
        "precision": "bfloat16",
        "attention": "sdpa",
        "batch_size": 1,
        "seed": 20260914,
        "enable_thinking": False,
        "do_sample": False,
        "max_input_tokens": 8192,
        "max_new_tokens": 2048,
        "context_limit": 40960,
        "stop_token_ids": [151645, 151643],
        "hard_inference_seconds": 1800,
        "copy_stop_reserve_seconds": 900,
        "max_provider_seconds": 10800,
        "max_session_usd": 3.0,
        "validation_inference": False,
        "training": False,
        "bulk_extraction": False,
        "automatic_retries": 0,
        "repair_generations": 0,
    }
    if any(plan.get(k) != v or type(plan.get(k)) is not type(v) for k, v in expected.items()):
        raise GateError("Execution plan differs from the fixed development-only benchmark")
    if set(plan.get("source_sha256", {})) != REQUIRED_SOURCES:
        raise GateError("Execution plan source set is incomplete")
    for relative, digest in plan["source_sha256"].items():
        if sha(REPO / relative) != digest:
            raise GateError(f"Execution source changed: {relative}")
    return plan


def load_package(prepared: Path, plan: dict):
    prepared = private(prepared)
    if sha(prepared / "plan.json") != PREPARED_PLAN_SHA:
        raise GateError("Prepared 40-report plan changed")
    package = read(prepared / "plan.json")
    if (package.get("execution_enabled") is not False or package.get("reports") != REPORTS
        or package.get("maximum_generations") != MAX_GENERATIONS
        or package.get("organizer_labels_in_model_inputs") is not False
        or package.get("validation_inference") is not False
        or package.get("prompt_sha256") != CONTROL_SHA
        or package.get("definitions_sha256") != DEFINITIONS_SHA):
        raise GateError("Prepared development package contract changed")
    for name, digest in package.get("files_sha256", {}).items():
        if name not in {"inputs.json", "prompt_inputs.json", "tokenized.json"} or sha(prepared / name) != digest:
            raise GateError(f"Prepared file changed: {name}")
    rows = read(prepared / "inputs.json")
    prompts = read(prepared / "prompt_inputs.json")
    tokens = read(prepared / "tokenized.json")
    if len(rows) != REPORTS or len(prompts) != REPORTS or len(tokens) != REPORTS:
        raise GateError("Exactly 40 prepared development reports required")
    if len({r.get("StudyInstanceUID") for r in rows}) != REPORTS:
        raise GateError("Prepared studies are not unique")
    allowed = {"case_index", "StudyInstanceUID", "Report", "split", "report_sha256"}
    label_names = {"ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
                   "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"}
    for index, (row, prompt, token) in enumerate(zip(rows, prompts, tokens)):
        if set(row) != allowed or set(row) & label_names or row["case_index"] != index \
                or row["split"] != "development" or not row["Report"]:
            raise GateError("Prepared input contains labels or invalid fields")
        if hashlib.sha256(row["Report"].encode()).hexdigest() != row["report_sha256"]:
            raise GateError("Prepared report hash mismatch")
        if prompt != {"case_index": index, "prompt": prompt["prompt"]} \
                or token.get("case_index") != index or prompt["prompt"] not in token.get("rendered_prompt", ""):
            raise GateError("Prompt/token case mapping changed")
        if token.get("input_tokens") != len(token.get("input_ids", [])) \
                or token["input_tokens"] > plan["max_input_tokens"] \
                or token["input_tokens"] + plan["max_new_tokens"] > plan["context_limit"]:
            raise GateError("Prepared token package exceeds fixed caps")
    return rows, prompts, tokens


def encode_checked(encoder, prompts, tokens, *, synthetic: bool):
    if synthetic:
        from v2_runtime import messages_for
    result = []
    for index, (prompt, expected) in enumerate(zip(prompts, tokens)):
        text = prompt["prompt"]
        if synthetic:
            rendered = encoder.processor.apply_chat_template(messages_for("qwen", text), tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            ids = encoder.tokenizer(rendered, add_special_tokens=False, truncation=False)["input_ids"]
            encoded = type("Encoded", (), {"rendered_prompt": rendered, "input_ids": list(ids),
                                             "input_tokens": len(ids)})()
        else:
            encoded = encoder.encode(text)
            if not hasattr(encoded, "inputs"):
                raise GateError("Real encoder did not produce backend tensors")
        if (encoded.rendered_prompt != expected["rendered_prompt"]
            or encoded.input_ids != expected["input_ids"]
            or encoded.input_tokens != expected["input_tokens"]):
            raise GateError(f"Rendered prompt/token mismatch at report {index}")
        result.append(encoded)
    return result


def validate_session(session, observation, stop, plan_sha: str, session_sha: str, plan: dict, *, now=None):
    required = {"schema_version", "approved_by_user", "execution_plan_sha256", "prepared_plan_sha256",
        "prompt_sha256", "pod_id", "region", "gpu_name", "gpu_count", "cloud_type", "container_disk_gb",
        "persistent_volume_gb", "network_volume_id", "compute_usd_per_hour", "storage_usd_per_hour",
        "maximum_usd", "t0", "hard_deadline", "shutdown_at"}
    if set(session) != required or session["schema_version"] != 1 or session["approved_by_user"] is not True:
        raise GateError("Fresh exact private user approval required")
    if (session["execution_plan_sha256"] != plan_sha
        or session["prepared_plan_sha256"] != PREPARED_PLAN_SHA
        or session["prompt_sha256"] != CONTROL_SHA):
        raise GateError("Approval does not bind this development run")
    if not isinstance(session["pod_id"], str) or not session["pod_id"] \
            or session["gpu_name"] not in ALLOWLIST or session["gpu_count"] != 1 \
            or session["cloud_type"] != "SECURE" or session["container_disk_gb"] != 80 \
            or session["persistent_volume_gb"] != 0 or session["network_volume_id"] is not None \
            or not isinstance(session["region"], str) or not session["region"]:
        raise GateError("Approved resource differs from the reviewed 48 GB Secure boundary")
    for key in ("compute_usd_per_hour", "storage_usd_per_hour", "maximum_usd"):
        if type(session[key]) not in (int, float) or not math.isfinite(session[key]) or session[key] < 0:
            raise GateError("Approved rate/budget is invalid")
    t0, deadline, shutdown = map(aware, (session["t0"], session["hard_deadline"], session["shutdown_at"]))
    now = now or datetime.now(timezone.utc)
    seconds = (deadline - t0).total_seconds()
    if (not 0 < seconds <= plan["max_provider_seconds"] or not t0 <= now < shutdown <= deadline
        or (deadline - shutdown).total_seconds() < 300 or session["maximum_usd"] > plan["max_session_usd"]
        or (session["compute_usd_per_hour"] + session["storage_usd_per_hour"]) * seconds / 3600
            > session["maximum_usd"]):
        raise GateError("Whole-session deadline, reserve or budget is invalid")
    observed_fields = {"pod_id", "provider_state", "observed_at", "region", "gpu_name", "gpu_count",
        "cloud_type", "container_disk_gb", "persistent_volume_gb", "network_volume_id",
        "actual_compute_usd_per_hour", "actual_storage_usd_per_hour", "current_total_usd_per_hour"}
    if set(observation) != observed_fields or observation["pod_id"] != session["pod_id"] \
            or observation["provider_state"] != "RUNNING":
        raise GateError("Fresh exact-pod RUNNING observation required")
    for key in ("region", "gpu_name", "gpu_count", "cloud_type", "container_disk_gb",
                "persistent_volume_gb", "network_volume_id"):
        if observation[key] != session[key] or type(observation[key]) is not type(session[key]):
            raise GateError("Observed resource differs from approval")
    for key in ("compute_usd_per_hour", "storage_usd_per_hour"):
        value = observation["actual_" + key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or value > session[key]:
            raise GateError("Observed rate is invalid or exceeds approval")
    total = observation["current_total_usd_per_hour"]
    if type(total) not in (int, float) or not math.isfinite(total) or total < 0 \
            or abs(total - observation["actual_compute_usd_per_hour"]
                   - observation["actual_storage_usd_per_hour"]) > 0.001:
        raise GateError("Observed total hourly billing is invalid")
    observed_at = aware(observation["observed_at"])
    if not t0 <= observed_at <= now or (now - observed_at).total_seconds() > 600:
        raise GateError("Provider observation is stale")
    expected_stop = {"pod_id", "execution_plan_sha256", "session_sha256", "shutdown_at", "checked_at",
                     "watchdog_pid", "watchdog_argv"}
    if (set(stop) != expected_stop or stop["pod_id"] != session["pod_id"]
        or stop["execution_plan_sha256"] != plan_sha or stop["session_sha256"] != session_sha
        or stop["shutdown_at"] != session["shutdown_at"]):
        raise GateError("Live shutdown receipt is missing or mismatched")
    if (now - aware(stop["checked_at"])).total_seconds() > 600 or aware(stop["checked_at"]) > now:
        raise GateError("Shutdown receipt is stale")
    ab_runtime().verify_watchdog(stop)
    room = min(plan["hard_inference_seconds"] - 1.0,
               (deadline - now).total_seconds() - plan["copy_stop_reserve_seconds"],
               (shutdown - now).total_seconds() - 1.0)
    if room < plan["hard_inference_seconds"] - 1.0:
        raise GateError("Insufficient time for inference and copy/stop reserve")
    return room


def check_live_step(session, stop, plan):
    now = datetime.now(timezone.utc)
    if now >= aware(session["shutdown_at"]) or (aware(session["hard_deadline"]) - now).total_seconds() \
            <= plan["copy_stop_reserve_seconds"]:
        raise TimeoutError("Development run reached its copy/stop reserve")
    ab_runtime().verify_watchdog(stop)


def run_worker(args) -> bool:
    plan = validate_plan(args.plan_sha256)
    rows, prompts, tokens = load_package(args.prepared, plan)
    synthetic = args.rehearsal
    session = observation = stop = None
    if not synthetic:
        session, observation, stop = map(private_json, (args.session, args.observation, args.shutdown_receipt))
        validate_session(session, observation, stop, args.plan_sha256, sha(args.session), plan)
    sys.path.insert(0, str(REPO / "analysis/report_labeling_llm_v2/scripts"))
    from v2_runtime import HFEncoder
    runtime = live_runtime()
    if synthetic:
        encoder = HFEncoder("qwen", args.cache)
    else:
        hardware = ab_runtime().verify_hardware(session["gpu_name"], runtime)
        from load_preflight import audit_cache
        audit = audit_cache(args.cache, "qwen")
        write_new(args.output / "cache_audit.json", audit)
        if not audit["complete"] or audit["revision"] != plan["model_revision"]:
            raise GateError("Pinned Qwen cache is incomplete")
        encoder = runtime.create_encoder(args.cache)
    ab_runtime().verify_effective_recipe(encoder.cfg, plan)
    encoded_rows = encode_checked(encoder, prompts, tokens, synthetic=synthetic)
    write_new(args.output / "input_preflight.json", {"synthetic": synthetic, "reports": REPORTS,
        "exact_prompt_and_token_parity": True, "prepared_plan_sha256": PREPARED_PLAN_SHA,
        "prompt_sha256": CONTROL_SHA, "input_tokens": [e.input_tokens for e in encoded_rows]})
    if synthetic:
        backend = ab_runtime().FakeBackend(encoder, args.fake_behavior)
    else:
        check_live_step(session, stop, plan)
        backend = runtime.create_backend(encoder, args.cache, session["gpu_name"])
        write_new(args.output / "model_preflight.json", {"hardware": hardware, "backend": backend.metadata,
            "eos_ids": backend.eos, "revision": plan["model_revision"]})
    result = {"status": "failed", "synthetic": synthetic, "planned": REPORTS, "attempted": 0,
              "completed": 0, "failed": 0, "unrun": REPORTS, "execution_plan_sha256": args.plan_sha256,
              "started_at": datetime.now(timezone.utc).isoformat()}
    started = time.monotonic()
    raw_written = parsed_written = 0
    loop_completed = False
    directory = args.output / "control-A"
    directory.mkdir(mode=0o700)
    try:
        with (directory / "attempts.jsonl").open("x") as attempts, \
             (directory / "raw.jsonl").open("x") as raw, \
             (directory / "parsed.jsonl").open("x") as parsed, \
             (directory / "technical_failures.jsonl").open("x") as failures:
            for stream in (attempts, raw, parsed, failures):
                os.chmod(stream.name, 0o600)
            for index, (row, encoded) in enumerate(zip(rows, encoded_rows)):
                if result["attempted"] >= MAX_GENERATIONS:
                    raise GateError("Forty-generation maximum reached")
                if not synthetic:
                    check_live_step(session, stop, plan)
                number = result["attempted"] + 1
                append(attempts, {"synthetic": synthetic, "attempt": number, "case_index": index,
                    "report_sha256": row["report_sha256"], "prompt_sha256": CONTROL_SHA,
                    "started_at": datetime.now(timezone.utc).isoformat()})
                result["attempted"] = number
                try:
                    generation = backend.generate(encoded)
                except BaseException as exc:
                    append(failures, {"synthetic": synthetic, "attempt": number,
                        "kind": "generation_exception", "error_type": type(exc).__name__})
                    result["failed"] += 1
                    raise
                record = {"synthetic": synthetic, "attempt": number, "case_index": index,
                    "StudyInstanceUID": row["StudyInstanceUID"], "report_sha256": row["report_sha256"],
                    "prompt_sha256": CONTROL_SHA, "rendered_prompt": encoded.rendered_prompt,
                    "input_ids": encoded.input_ids, "input_tokens": encoded.input_tokens,
                    "generation": generation, "gpu_peak_allocated_gib": backend.peak_gib()}
                append(raw, record)
                raw_written += 1
                result["completed"] += generation.get("generation_status") == "completed"
                try:
                    primary, secondary = runtime.validate_generation(generation, backend, encoded, row["Report"])
                except BaseException as exc:
                    append(failures, {"synthetic": synthetic, "attempt": number,
                        "kind": "integrity_or_incomplete_generation", "error_type": type(exc).__name__})
                    result["failed"] += 1
                    raise
                append(parsed, {"synthetic": synthetic, "attempt": number, "prompt_sha256": CONTROL_SHA,
                    "primary_v2": primary, "secondary_v1": secondary,
                    "raw_sha256": hashlib.sha256(json.dumps(record, sort_keys=True,
                        ensure_ascii=False, allow_nan=False).encode()).hexdigest()})
                parsed_written += 1
                errors = [r["status"] for r in primary["rows"] if r["status"] != "valid"]
                if errors:
                    append(failures, {"synthetic": synthetic, "attempt": number,
                        "kind": "parser_technical_failure", "condition_statuses": errors})
        write_new(args.output / "control-A.complete.json", {"synthetic": synthetic,
            "completed_generations": REPORTS, "finished_at": datetime.now(timezone.utc).isoformat()})
        loop_completed = True
    except BaseException as exc:
        result["failure_type"] = type(exc).__name__
    finally:
        complete = (loop_completed and result["attempted"] == result["completed"] == REPORTS
                    and result["failed"] == 0 and raw_written == parsed_written == REPORTS
                    and "failure_type" not in result)
        result.update(status="completed" if complete else "failed", raw_records_written=raw_written,
            parsed_records_written=parsed_written, unrun=REPORTS-result["attempted"],
            elapsed_seconds=time.monotonic()-started, peak_gpu_allocated_gib=backend.peak_gib(),
            finished_at=datetime.now(timezone.utc).isoformat(),
            artifacts_sha256={str(p.relative_to(args.output)): sha(p) for p in sorted(args.output.rglob("*"))
                              if p.is_file() and p.name != "session_result.json"})
        write_new(args.output / "session_result.json", result)
    return result["status"] == "completed"


def verify_child(args):
    if args.worker_fd is None or args.worker_fd < 3 or not stat.S_ISFIFO(os.fstat(args.worker_fd).st_mode):
        raise GateError("Worker requires active supervisor pipe")
    with os.fdopen(args.worker_fd) as stream:
        message = json.loads(stream.read(4096))
    start_path, dispatch_path = args.output / "supervisor_start.json", args.output / "supervisor_dispatch.json"
    start, dispatch = read(start_path), read(dispatch_path)
    if (message.get("parent_pid") != os.getppid() or start.get("parent_pid") != os.getppid()
        or message.get("plan_sha256") != args.plan_sha256 or start.get("plan_sha256") != args.plan_sha256
        or hashlib.sha256(message.get("nonce", "").encode()).hexdigest() != dispatch.get("nonce_sha256")
        or dispatch.get("start_sha256") != sha(start_path)
        or start.get("prepared_plan_sha256") != PREPARED_PLAN_SHA
        or (args.output / "session_result.json").exists()):
        raise GateError("Worker lacks active parent attestation")
    if not args.rehearsal and (start.get("session_sha256") != sha(args.session)
        or start.get("observation_sha256") != sha(args.observation)
        or start.get("shutdown_receipt_sha256") != sha(args.shutdown_receipt)):
        raise GateError("Private receipts changed after launch")


def launch(args, timeout):
    sys.path.insert(0, str(REPO / "analysis/report_labeling_llm_v2/scripts"))
    from run_smoke import supervise
    if args.output.exists():
        raise FileExistsError("Never overwrite or resume a prior output")
    args.output.mkdir(mode=0o700, parents=True)
    write_new(args.output / "supervisor_start.json", {"synthetic": args.rehearsal,
        "parent_pid": os.getpid(), "plan_sha256": args.plan_sha256,
        "prepared_plan_sha256": PREPARED_PLAN_SHA,
        "session_sha256": None if args.rehearsal else sha(args.session),
        "observation_sha256": None if args.rehearsal else sha(args.observation),
        "shutdown_receipt_sha256": None if args.rehearsal else sha(args.shutdown_receipt),
        "started_at": datetime.now(timezone.utc).isoformat()})
    nonce = secrets.token_hex(32)
    write_new(args.output / "supervisor_dispatch.json", {"start_sha256": sha(args.output / "supervisor_start.json"),
        "nonce_sha256": hashlib.sha256(nonce.encode()).hexdigest()})
    pipe_r, pipe_w = os.pipe()
    command = [sys.executable, str(HERE), "--worker", "--prepared", str(args.prepared),
        "--cache", str(args.cache), "--output", str(args.output), "--plan-sha256", args.plan_sha256,
        "--worker-fd", str(pipe_r)]
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
        "partial_artifacts_sha256": {str(p.relative_to(args.output)): sha(p)
            for p in sorted(args.output.rglob("*")) if p.is_file() and p.name != "supervisor_result.json"}})
    return okay, status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("prepared", "cache", "output", "session", "observation", "shutdown-receipt"):
        parser.add_argument("--" + flag, type=Path)
    parser.add_argument("--plan-sha256", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--rehearsal", action="store_true")
    group.add_argument("--run", action="store_true")
    parser.add_argument("--fake-behavior", choices=("normal", "hang", "raise", "incomplete", "schema_error"),
                        default="normal", help=argparse.SUPPRESS)
    parser.add_argument("--rehearsal-timeout", type=float, default=60.0, help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.prepared is None or args.cache is None:
        parser.error("--prepared and --cache are required")
    args.prepared, args.cache = map(Path, (args.prepared, args.cache))
    plan = validate_plan(args.plan_sha256)
    if args.worker:
        if args.output is None or not (args.run or args.rehearsal):
            raise GateError("Worker requires complete parent dispatch")
        args.output = private(args.output)
        verify_child(args)
        return 0 if run_worker(args) else 1
    rows, prompts, tokens = load_package(args.prepared, plan)
    if not args.run and not args.rehearsal:
        sys.path.insert(0, str(REPO / "analysis/report_labeling_llm_v2/scripts"))
        from v2_runtime import HFEncoder
        encoder = HFEncoder("qwen", args.cache)
        encode_checked(encoder, prompts, tokens, synthetic=True)
        print(json.dumps({"status": "DRY_RUN_PASS", "reports": len(rows), "generations_executed": 0,
            "execution_plan_sha256": args.plan_sha256, "prepared_plan_sha256": PREPARED_PLAN_SHA,
            "organizer_labels_in_model_inputs": False, "exact_prompt_and_token_parity": True}, indent=2))
        return 0
    if args.output is None:
        parser.error("--output is required")
    args.output = private(args.output)
    if args.rehearsal:
        if not 0 < args.rehearsal_timeout <= 60:
            raise GateError("Synthetic timeout must be 0-60 seconds")
        timeout = args.rehearsal_timeout
    else:
        if any(x is None for x in (args.session, args.observation, args.shutdown_receipt)):
            raise GateError("Real run needs fresh approval, RUNNING observation and shutdown receipt")
        args.session, args.observation, args.shutdown_receipt = map(private,
            (args.session, args.observation, args.shutdown_receipt))
        timeout = validate_session(private_json(args.session), private_json(args.observation),
            private_json(args.shutdown_receipt), args.plan_sha256, sha(args.session), plan)
    okay, status = launch(args, timeout)
    print(json.dumps({"status": status, "synthetic": args.rehearsal,
        "worker_exit_success": okay, "output": str(args.output)}, indent=2))
    return 0 if okay else 1


if __name__ == "__main__":
    raise SystemExit(main())
