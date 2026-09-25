"""Operator-supervised Qwen extraction for one already-running, approved RunPod."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import secrets
import stat
import subprocess
import sys
import time

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
REPO = ROOT.parents[1]
POLICY_PATH = ROOT / "configs/session_policy.json"
PLAN_PATH = ROOT / "configs/execution_plan.json"
MANIFEST_PATH = ROOT / "configs/source_manifest.json"
PREPARED_SHA = "c4586bb0c65163244dac80344ea96f691493a1b89a5911eefcba54f2384c74fc"
REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"
ORDER = ["control-1", "candidate-1", "candidate-2", "control-2"]
EXPECTED_POD_NAME = "low_brown_viper"
EXPECTED_POD_ID_SHA256 = hashlib.sha256(b"rqnenq3mpu0i2g").hexdigest()
EXPECTED_GPU = "NVIDIA RTX A6000"
EXPECTED_CLOUD = "SECURE"
EXPECTED_REGION = "EU-SE-1"
EXPECTED_DISK_GB = 80
EXPECTED_GPU_COUNT = 1
EXPECTED_VRAM_GB = 48


class SessionError(RuntimeError):
    pass


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load frozen module {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def qwen_runtime():
    return load_module("qwen_operator_existing_runtime", REPO / "analysis/qwen_execution_runtime_alt_v1/scripts/qwen_live.py")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def write_new(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def append_durable(stream, value: dict) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def aware(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise SessionError("Timezone-aware session timestamps are required")
    return parsed.astimezone(timezone.utc)


def private_json(path: Path) -> dict:
    info = Path(path).lstat()
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
        raise PermissionError("Operator session file must be regular and mode 0600")
    return read_json(path)


def validate_session(session: dict, *, now: datetime | None = None) -> dict:
    required = {"schema_version", "pod_name", "pod_id", "system_ram_gb", "provider_state", "observed_at",
        "cloud_type", "gpu_name", "gpu_count", "gpu_vram_gb", "region", "container_disk_gb",
        "persistent_volume_gb", "network_volume_id", "compute_usd_per_hour",
        "storage_usd_per_hour", "maximum_usd", "t0", "hard_deadline", "shutdown_at",
        "execution_plan_sha256", "prepared_plan_sha256", "model_revision", "prior_session_cost_usd"}
    if not isinstance(session, dict) or set(session) != required or session["schema_version"] != 1:
        raise SessionError("Operator session facts have missing or unexpected fields")
    exact = {"pod_name": EXPECTED_POD_NAME, "system_ram_gb": 50, "provider_state": "RUNNING",
        "cloud_type": EXPECTED_CLOUD, "gpu_name": EXPECTED_GPU, "gpu_count": 1,
        "gpu_vram_gb": 48, "region": EXPECTED_REGION, "container_disk_gb": 80,
        "persistent_volume_gb": 0, "network_volume_id": None, "prepared_plan_sha256": PREPARED_SHA,
        "model_revision": REVISION}
    for key, value in exact.items():
        if session.get(key) != value or type(session.get(key)) is not type(value):
            raise SessionError(f"Approved resource or scientific identity mismatch: {key}")
    if not isinstance(session["pod_id"], str) or hashlib.sha256(session["pod_id"].encode()).hexdigest() != EXPECTED_POD_ID_SHA256:
        raise SessionError("Approved pod ID differs from the selected low_brown_viper pod")
    for key, ceiling in (("compute_usd_per_hour", 0.84), ("storage_usd_per_hour", 0.012), ("maximum_usd", 3.0)):
        value = session[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= ceiling:
            raise SessionError(f"Approved cost ceiling exceeded: {key}")
    if session["compute_usd_per_hour"] <= 0 or session["maximum_usd"] <= 0:
        raise SessionError("Positive provider rates and budget are required")
    if (session["compute_usd_per_hour"] + session["storage_usd_per_hour"]) * 3 > session["maximum_usd"]:
        raise SessionError("Three-hour maximum charge exceeds approved budget")
    previous = session["prior_session_cost_usd"]
    if type(previous) not in (int, float) or not math.isfinite(previous) or previous < 0:
        raise SessionError("Prior spend must remain visible and nonnegative")
    plan_hash = session["execution_plan_sha256"]
    if not isinstance(plan_hash, str) or len(plan_hash) != 64 or any(c not in "0123456789abcdef" for c in plan_hash):
        raise SessionError("New execution plan hash is invalid")
    t0, deadline, shutdown = aware(session["t0"]), aware(session["hard_deadline"]), aware(session["shutdown_at"])
    if deadline != t0 + timedelta(hours=3) or shutdown != t0 + timedelta(minutes=175):
        raise SessionError("Fixed T0+180 / T0+175 deadline binding mismatch")
    now = now or datetime.now(timezone.utc)
    observed = aware(session["observed_at"])
    if not t0 <= now < deadline:
        raise SessionError("Session has not started or its immutable deadline has expired")
    if not t0 <= observed <= now:
        raise SessionError("Running-pod observation timestamp is outside the approved session/deadline")
    if (now - observed).total_seconds() > 600:
        raise SessionError("Running-pod operator observation is stale or outside the session")
    return {"t0": t0, "deadline": deadline, "shutdown_at": shutdown,
            "inference_cutoff": deadline - timedelta(seconds=900)}


def validate_plan(expected_sha256: str) -> dict:
    plan = read_json(PLAN_PATH)
    manifest = read_json(MANIFEST_PATH)
    if sha(PLAN_PATH) != expected_sha256 or plan.get("prepared_plan_sha256") != PREPARED_SHA:
        raise SessionError("New operator execution-plan hash does not match")
    if sha(MANIFEST_PATH) != plan.get("source_manifest_sha256"):
        raise SessionError("Operator source-manifest hash differs from the execution plan")
    if sha(POLICY_PATH) != plan.get("session_policy_sha256"):
        raise SessionError("Running-session policy differs from the execution plan")
    if plan.get("real_execution_enabled") is not False:
        raise SessionError("The operator runtime must remain disabled by default")
    expected_resource = {"pod_name": EXPECTED_POD_NAME,
        "pod_id_sha256": EXPECTED_POD_ID_SHA256, "cloud_type": EXPECTED_CLOUD,
        "region": EXPECTED_REGION, "gpu_name": EXPECTED_GPU, "gpu_count": 1,
        "gpu_vram_gb": 48, "system_ram_gb": 50, "container_disk_gb": 80,
        "persistent_volume_gb": 0, "network_volume_id": None,
        "compute_usd_per_hour": 0.53, "storage_usd_per_hour": 0.011,
        "maximum_session_usd": 3.0, "maximum_provider_seconds": 10800,
        "shutdown_at_seconds": 10500}
    if plan.get("resource") != expected_resource or plan.get("historical_cost_usd_separate") != 0.41:
        raise SessionError("New running-session resource boundary changed")
    for relative, expected in manifest.get("source_sha256", {}).items():
        if sha(REPO / relative) != expected:
            raise SessionError(f"Bound execution source changed: {relative}")
    if (plan.get("scope") != {"model_id": "Qwen/Qwen3-14B", "model_revision": REVISION,
            "run_order": ORDER, "reports_per_block": 5, "maximum_generations": 20,
            "precision": "bfloat16", "attention": "sdpa", "batch_size": 1,
            "training": False, "validation_inference": False, "bulk_extraction": False,
            "retry_or_repair": False}):
        raise SessionError("Execution-plan scientific scope drift")
    return plan


def require_time_for_run(session: dict, *, now: datetime | None = None) -> float:
    times = validate_session(session, now=now)
    now = now or datetime.now(timezone.utc)
    room = (times["inference_cutoff"] - now).total_seconds()
    timeout = min(1800.0, room)
    if timeout < 1800:
        raise TimeoutError("Not enough time remains for the bounded inference cap plus 15-minute copy/stop reserve")
    return timeout


def observe_hardware(runtime) -> dict:
    # Preserve frozen package pins while accepting Torch's explicit cu128 local tag.
    from v2_runtime import check_versions
    import torch
    versions = check_versions()
    expected = runtime.runtime_config()["required_versions"]
    normalized = dict(versions)
    if normalized.get("torch") == "2.8.0+cu128":
        normalized["torch"] = "2.8.0"
    if normalized != expected:
        raise SessionError("Pinned dependency versions differ")
    def query(field):
        result = subprocess.run(["nvidia-smi", "--query-gpu=" + field, "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=15)
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    names, drivers = query("name"), query("driver_version")
    if len(drivers) != 1:
        raise SessionError("Single GPU driver provenance required")
    runtime.verify_reported_hardware(names, torch.cuda.device_count(), torch.version.cuda,
        torch.__version__, torch.cuda.is_available(),
        torch.cuda.is_bf16_supported(including_emulation=False),
        torch.cuda.mem_get_info(0)[0] / 2**30, EXPECTED_GPU)
    return dict(gpu_names=names, gpu_driver_version=drivers[0],
        cuda_version=torch.version.cuda, torch_version=torch.__version__,
        package_versions=versions, native_bf16=True)


def verify_hardware(session: dict) -> dict:
    # Check for other work BEFORE this process initializes its own CUDA context.
    # nvidia-smi process IDs may be host PIDs rather than container PIDs.
    process = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        check=True, capture_output=True, text=True, timeout=15)
    existing = [line for line in process.stdout.splitlines() if line.strip()]
    if existing:
        raise SessionError("Another GPU workload is already using this device")
    runtime = qwen_runtime()
    observed = observe_hardware(runtime)
    if observed["gpu_names"] != [EXPECTED_GPU]:
        raise SessionError("Visible GPU identity differs from the approved RTX A6000")
    query = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15)
    rows = [line.strip().split(",") for line in query.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 2:
        raise SessionError("Could not verify exactly one GPU and its free memory")
    total_mib, free_mib = map(int, rows[0])
    # nvidia-smi reports MiB; the approved 48 GB capacity is decimal bytes.
    if total_mib * 1024**2 < 48_000_000_000 or free_mib < 36 * 1024:
        raise SessionError("RTX A6000 requires at least 36 GiB free VRAM")
    observed.update(total_gpu_mib=total_mib, free_gpu_mib=free_mib, existing_gpu_processes=existing)
    return observed


def run_worker(prepared: Path, cache: Path, output: Path, session_path: Path) -> bool:
    runtime = qwen_runtime()
    session = private_json(session_path)
    require_time_for_run(session)
    _, rows = runtime.check_plan(prepared, PREPARED_SHA)
    if len(rows) != 5:
        raise SessionError("Prepared package must contain exactly five development reports")
    manifest = read_json(MANIFEST_PATH)
    for relative, expected in manifest.get("prepared_file_sha256", {}).items():
        if sha(prepared / relative) != expected:
            raise SessionError(f"Prepared five-report package changed: {relative}")
    hw = verify_hardware(session)
    from load_preflight import audit_cache
    audit = audit_cache(cache, "qwen")
    write_new(output / "cache_audit.json", audit)
    if not audit["complete"] or audit["revision"] != REVISION:
        raise SessionError("Pinned Qwen cache is incomplete or revision-mismatched")
    encoder = runtime.create_encoder(cache)
    arms = {arm: runtime.verify_prepared_tokens(prepared, arm, rows, encoder)
        for arm in ("control", "candidate")}
    write_new(output / "tokenizer_preflight.json", {"model_revision": REVISION,
        "prepared_plan_sha256": PREPARED_SHA, "exact_prepared_parity": True,
        "input_tokens": {arm: [item.input_tokens for item in values] for arm, values in arms.items()}})
    require_time_for_run(session)
    backend = runtime.create_backend(encoder, cache, EXPECTED_GPU)
    write_new(output / "model_preflight.json", {"hardware": hw, "backend": backend.metadata,
        "eos_ids": backend.eos, "attention": "sdpa", "revision": REVISION})
    result = {"status": "failed", "pod_id": session["pod_id"], "gpu_name": EXPECTED_GPU,
        "execution_plan_sha256": session["execution_plan_sha256"], "prepared_plan_sha256": PREPARED_SHA,
        "model_revision": REVISION, "run_order": [], "planned": 20, "attempted": 0,
        "completed": 0, "failed": 0, "unrun": 20, "training_steps": 0,
        "started_at": datetime.now(timezone.utc).isoformat()}
    started = time.monotonic()
    try:
        for run_key in ORDER:
            if (aware(session["hard_deadline"]) - datetime.now(timezone.utc)).total_seconds() < 900:
                raise TimeoutError("Fifteen-minute hard-deadline shutdown reserve reached")
            arm = run_key.split("-")[0]
            run_dir = output / run_key
            run_dir.mkdir(mode=0o700)
            with (run_dir / "attempts.jsonl").open("x") as attempts, \
                 (run_dir / "raw.jsonl").open("x") as raw_stream, \
                 (run_dir / "parsed.jsonl").open("x") as parsed_stream, \
                 (run_dir / "technical_failures.jsonl").open("x") as failures:
                for stream in (attempts, raw_stream, parsed_stream, failures):
                    os.chmod(stream.name, 0o600)
                for index, (row, encoded) in enumerate(zip(rows, arms[arm])):
                    if result["attempted"] >= 20:
                        raise SessionError("Twenty-generation maximum reached")
                    if (aware(session["hard_deadline"]) - datetime.now(timezone.utc)).total_seconds() < 900:
                        raise TimeoutError("Hard shutdown reserve reached before generation")
                    number = result["attempted"] + 1
                    append_durable(attempts, {"attempt": number, "run_key": run_key,
                        "case_index": index, "pod_id": session["pod_id"],
                        "started_at": datetime.now(timezone.utc).isoformat(),
                        "report_sha256": row["report_sha256"]})
                    result["attempted"] = number
                    try:
                        generation = backend.generate(encoded)
                    except BaseException as exc:
                        append_durable(failures, {"attempt": number, "kind": "generation_exception",
                            "error_type": type(exc).__name__, "recorded_at": datetime.now(timezone.utc).isoformat()})
                        result["failed"] += 1
                        raise
                    raw_record = {"attempt": number, "run_key": run_key, "case_index": index,
                        "StudyInstanceUID": row["StudyInstanceUID"], "report_sha256": row["report_sha256"],
                        "rendered_prompt": encoded.rendered_prompt, "input_ids": encoded.input_ids,
                        "input_tokens": encoded.input_tokens, "generation": generation,
                        "gpu_peak_allocated_gib": backend.peak_gib()}
                    append_durable(raw_stream, raw_record)
                    result["completed"] += int(generation.get("generation_status") == "completed")
                    try:
                        primary, secondary = runtime.validate_generation(generation, backend, encoded, row["Report"])
                        append_durable(parsed_stream, {"attempt": number, "primary_v2": primary,
                            "secondary_v1": secondary, "raw_sha256": hashlib.sha256(json.dumps(raw_record,
                            sort_keys=True, ensure_ascii=False).encode()).hexdigest()})
                    except BaseException as exc:
                        result["failed"] += 1
                        append_durable(failures, {"attempt": number, "kind": "technical_validation_failure",
                            "error_type": type(exc).__name__, "generation_status": generation.get("generation_status"),
                            "recorded_at": datetime.now(timezone.utc).isoformat()})
                        raise
            write_new(output / f"{run_key}.complete.json", {"run_key": run_key,
                "completed_generations": 5, "finished_at": datetime.now(timezone.utc).isoformat()})
            result["run_order"].append(run_key)
            result["unrun"] = 20 - result["attempted"]
        result["status"] = "completed" if result["completed"] == 20 and result["failed"] == 0 else "failed"
    except BaseException as exc:
        result.update(status="failed", failure_type=type(exc).__name__,
            failure_recorded_at=datetime.now(timezone.utc).isoformat())
        if not isinstance(exc, Exception):
            raise
    finally:
        result.update(elapsed_seconds=time.monotonic() - started,
            finished_at=datetime.now(timezone.utc).isoformat(), peak_gpu_allocated_gib=backend.peak_gib(),
            unrun=20-result["attempted"], artifacts_sha256={str(p.relative_to(output)): sha(p)
            for p in sorted(output.rglob("*")) if p.is_file() and p.name != "session_result.json"})
        write_new(output / "session_result.json", result)
    return result["status"] == "completed"


def launch(prepared: Path, cache: Path, output: Path, session_path: Path) -> dict:
    session = private_json(session_path)
    validate_plan(session["execution_plan_sha256"])
    timeout = require_time_for_run(session)
    if output.exists():
        raise FileExistsError("Refusing to overwrite or resume a previous run")
    output.mkdir(parents=True, mode=0o700)
    start = datetime.now(timezone.utc)
    receipt = {"status": "started", "pod_id": session["pod_id"],
        "t0": session["t0"], "hard_deadline": session["hard_deadline"],
        "shutdown_at": session["shutdown_at"], "execution_plan_sha256": session["execution_plan_sha256"],
        "session_file_sha256": sha(session_path), "supervisor_timeout_seconds": timeout,
        "parent_pid": os.getpid(), "started_at": start.isoformat()}
    write_new(output / "supervisor_start.json", receipt)
    nonce = secrets.token_hex(32)
    dispatch = {"supervisor_start_sha256": sha(output / "supervisor_start.json"),
        "nonce_sha256": hashlib.sha256(nonce.encode()).hexdigest()}
    write_new(output / "supervisor_dispatch.json", dispatch)
    pipe_r, pipe_w = os.pipe()
    command = [sys.executable, str(HERE), "--worker", "--prepared", str(prepared),
        "--cache", str(cache), "--output", str(output), "--session", str(session_path),
        "--worker-fd", str(pipe_r)]
    sys.path.insert(0, str(REPO / "analysis/report_labeling_llm_v2/scripts"))
    from run_smoke import supervise
    with os.fdopen(pipe_w, "w") as writer:
        json.dump({"parent_pid": os.getpid(), "nonce": nonce,
            "execution_plan_sha256": session["execution_plan_sha256"]}, writer)
    try:
        okay, status = supervise(command, timeout, pass_fds=(pipe_r,))
    finally:
        os.close(pipe_r)
    result = {"status": status, "worker_exit_success": okay, "pod_id": session["pod_id"],
        "hard_timeout_seconds": timeout, "partial_artifacts_sha256": {str(p.relative_to(output)): sha(p)
        for p in sorted(output.rglob("*")) if p.is_file() and p.name != "supervisor_result.json"}}
    write_new(output / "supervisor_result.json", result)
    return result


def worker(args) -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--worker-fd", type=int, required=True)
    options = parser.parse_args(args)
    if options.worker_fd < 3 or not stat.S_ISFIFO(os.fstat(options.worker_fd).st_mode):
        raise PermissionError("Worker requires the active supervisor pipe")
    with os.fdopen(options.worker_fd) as stream:
        dispatch = json.loads(stream.read(4096))
    start = read_json(options.output / "supervisor_start.json")
    recorded_dispatch = read_json(options.output / "supervisor_dispatch.json")
    if (dispatch.get("parent_pid") != os.getppid() or start.get("parent_pid") != os.getppid()
        or hashlib.sha256(dispatch.get("nonce", "").encode()).hexdigest() != recorded_dispatch.get("nonce_sha256")
        or recorded_dispatch.get("supervisor_start_sha256") != sha(options.output / "supervisor_start.json")
        or sha(options.session) != start.get("session_file_sha256")
        or (options.output / "session_result.json").exists()):
        raise PermissionError("Worker is not bound to the active supervised run")
    okay = run_worker(options.prepared, options.cache, options.output, options.session)
    return 0 if okay else 1


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--session", type=Path)
    parser.add_argument("--plan-sha256", required=False)
    parser.add_argument("--run", action="store_true", help="Execute the exact approved 20-generation smoke")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return worker(sys.argv[1:])
    if not args.run or any(item is None for item in (args.prepared, args.cache, args.output, args.session)):
        raise SystemExit("Explicit --run and all exact local paths are required")
    session = private_json(args.session)
    if args.plan_sha256 != session.get("execution_plan_sha256"):
        raise PermissionError("Execution plan hash does not match the operator session")
    validate_plan(args.plan_sha256)
    result = launch(args.prepared, args.cache, args.output, args.session)
    return 0 if result["worker_exit_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
