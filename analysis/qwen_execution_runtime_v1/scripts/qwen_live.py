"""Separately versioned, fail-closed Qwen smoke execution runtime."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
QWEN_RUNTIME = ROOT.parent / "qwen_runtime_v1"
QWEN_EXECUTION = ROOT.parent / "qwen_execution_v1"
V2 = ROOT.parent / "report_labeling_llm_v2"
PREPARED_PLAN_SHA256 = "c4586bb0c65163244dac80344ea96f691493a1b89a5911eefcba54f2384c74fc"
PROPOSAL_SHA256 = "61d0dfe522f4ad4e4d00fef86532bc55f50c989a2495c6e9d6637be036165b12"
QWEN_STOP_IDS = [151645, 151643]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


frozen = load_module("qwen_live_frozen_runtime", QWEN_RUNTIME / "scripts/runtime_qwen.py")
resource_gate = load_module("qwen_live_resource_gate", QWEN_EXECUTION / "scripts/resource_gate.py")
sys.path.insert(0, str(V2 / "scripts"))
from run_smoke import supervise  # Frozen process-group hard timeout and kill/reap helper.


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=frozen.core.no_duplicates)


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("x") as f:
        os.chmod(path, 0o600)
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def append_durable(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def execution_plan():
    return read(ROOT / "configs/execution_plan.json")


def runtime_config():
    return read(ROOT / "configs/runtime.json")


def validate_execution_plan(plan_path, expected_sha256):
    plan_path = Path(plan_path)
    if sha(plan_path) != expected_sha256:
        raise PermissionError("Execution plan SHA-256 mismatch")
    expected = execution_plan()
    if read(plan_path) != expected:
        raise ValueError("Execution plan differs from the committed fixed-scope plan")
    if sha(QWEN_EXECUTION / "configs/resource_proposal.json") != expected["resource_proposal_sha256"]:
        raise PermissionError("Frozen resource proposal SHA-256 mismatch")
    if sha(ROOT / "configs/runtime.json") != expected["runtime_policy_sha256"]:
        raise PermissionError("Execution policy differs from the approved plan")
    if sha(Path(__file__)) != expected["runtime_source_sha256"]:
        raise PermissionError("Execution source differs from the approved plan")
    if sha(QWEN_EXECUTION / "scripts/resource_gate.py") != expected["resource_gate_sha256"]:
        raise PermissionError("Frozen authorization gate source changed")
    if sha(QWEN_EXECUTION / "scripts/stop_watchdog.py") != expected["watchdog_source_sha256"]:
        raise PermissionError("Frozen watchdog source changed")
    return expected


def private_file(path, label):
    import stat
    p = Path(path).resolve()
    mode = stat.S_IMODE(p.stat().st_mode)
    if not stat.S_ISREG(p.stat().st_mode) or mode != 0o600:
        raise PermissionError(f"{label} must be a regular file with mode 0600")
    return p


def live_gate(plan_path, plan_sha256, authorization, watchdog_receipt, stop_preflight, current=None):
    """All authorization, exact-pod and watchdog checks happen before cache/model access."""
    cfg = runtime_config()
    if not cfg["execution_enabled"]:
        raise PermissionError("Real execution is disabled in the committed runtime policy")
    validate_execution_plan(plan_path, plan_sha256)
    grant_path = private_file(authorization, "Authorization")
    watchdog_path = private_file(watchdog_receipt, "Watchdog receipt")
    preflight_path = private_file(stop_preflight, "Stop preflight receipt")
    grant = validate_user_approval(grant_path, plan_sha256, current=current)
    resource_gate.verify_watchdog(watchdog_path, grant,
        QWEN_EXECUTION / "scripts/stop_watchdog.py", preflight_path, current=current)
    return grant


def validate_user_approval(path, execution_plan_sha256, current=None):
    """Validate this stage grant and bind the new execution plan as well as frozen inputs."""
    from datetime import timedelta
    import math
    path = private_file(path, "Authorization")
    grant = read(path)
    proposal_path = QWEN_EXECUTION / "configs/resource_proposal.json"
    proposal = read(proposal_path)
    cfg = runtime_config()
    required = {"approved", "approval_reference", "plan_sha256", "prepared_plan_sha256",
        "proposal_sha256", "approved_at", "provider_start_requested_at", "provider_deadline",
        "watchdog_stop_at", "maximum_usd", "actual_compute_usd_per_hour",
        "actual_storage_usd_per_hour", "pod_id", "gpu_name", "gpu_count", "cloud_type",
        "container_disk_gb", "persistent_volume_gb", "network_volume_id"}
    if set(grant) != required or grant["approved"] is not True or not str(grant["approval_reference"]).strip():
        raise PermissionError("Exact private user approval record required")
    if grant["plan_sha256"] != execution_plan_sha256 or grant["prepared_plan_sha256"] != PREPARED_PLAN_SHA256:
        raise PermissionError("Approval is not bound to these exact execution and prepared plans")
    if grant["proposal_sha256"] != PROPOSAL_SHA256 or sha(proposal_path) != PROPOSAL_SHA256:
        raise PermissionError("Approval is not bound to the pinned proposal")
    if (grant["gpu_name"] != proposal["gpu_name"] or type(grant["gpu_count"]) is not int or grant["gpu_count"] != 1
        or grant["cloud_type"] != "SECURE" or type(grant["container_disk_gb"]) is not int or grant["container_disk_gb"] != 80
        or type(grant["persistent_volume_gb"]) is not int or grant["persistent_volume_gb"] != 0
        or grant["network_volume_id"] is not None or not str(grant["pod_id"]).strip()):
        raise PermissionError("Approval resource identity or storage differs from proposal")
    current = current or datetime.now(timezone.utc)
    if current > resource_gate.parse_time(proposal["quote_valid_until"]):
        raise PermissionError("Resource quote expired")
    approved = resource_gate.parse_time(grant["approved_at"])
    started = resource_gate.parse_time(grant["provider_start_requested_at"])
    deadline = resource_gate.parse_time(grant["provider_deadline"])
    stop_at = resource_gate.parse_time(grant["watchdog_stop_at"])
    if (not approved <= started <= current < stop_at < deadline or
        stop_at != deadline - timedelta(seconds=300) or
        not 0 < (deadline-started).total_seconds() <= cfg["maximum_provider_seconds"] or current >= stop_at):
        raise PermissionError("Approval timestamps or fixed provider/shutdown window are invalid")
    amounts = [grant[k] for k in ("maximum_usd", "actual_compute_usd_per_hour", "actual_storage_usd_per_hour")]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in amounts):
        raise PermissionError("Invalid approved cost values")
    if (not 0 < grant["maximum_usd"] <= cfg["maximum_approved_usd"] or
        grant["actual_compute_usd_per_hour"] > proposal["compute_usd_per_hour_ceiling"] or
        grant["actual_storage_usd_per_hour"] > proposal["storage_usd_per_hour_ceiling"] or
        sum(amounts[1:])*(deadline-started).total_seconds()/3600 > grant["maximum_usd"]):
        raise PermissionError("Approved rates or budget exceed exact proposal boundary")
    return grant


def verify_live_watchdog(watchdog_receipt, grant, stop_preflight):
    """Re-check process PID, command, environment and bound receipts between generations."""
    return resource_gate.verify_watchdog(watchdog_receipt, grant,
        QWEN_EXECUTION / "scripts/stop_watchdog.py", stop_preflight)


def pre_generation_gate(watchdog_receipt, grant, stop_preflight, started, current=None):
    check_time_budget(grant, started, current=current, minimum_remaining_seconds=330)
    verify_live_watchdog(watchdog_receipt, grant, stop_preflight)
    # Account for time spent validating the live process before entering CUDA generation.
    check_time_budget(grant, started, current=current, minimum_remaining_seconds=330)


def verify_worker_attestation(fd, output, execution_plan_sha256, prepared_plan_sha256,
                              authorization, watchdog_receipt, stop_preflight):
    import stat
    if fd is None or fd < 3 or not stat.S_ISFIFO(os.fstat(fd).st_mode):
        raise PermissionError("Supervised worker requires inherited parent pipe")
    with os.fdopen(fd) as stream:
        message = json.loads(stream.read(8192))
    output = Path(output).resolve()
    start_path, dispatch_path = output / "supervisor_start.json", output / "supervisor_dispatch.json"
    start, dispatch = read(start_path), read(dispatch_path)
    if (message.get("parent_pid") != os.getppid() or start.get("parent_pid") != os.getppid()
        or message.get("execution_plan_sha256") != execution_plan_sha256
        or message.get("prepared_plan_sha256") != prepared_plan_sha256
        or digest(message.get("nonce")) != dispatch.get("nonce_sha256")
        or dispatch.get("supervisor_start_sha256") != sha(start_path)
        or start.get("execution_plan_sha256") != execution_plan_sha256
        or start.get("prepared_plan_sha256") != prepared_plan_sha256
        or start.get("authorization_sha256") != sha(authorization)
        or start.get("watchdog_receipt_sha256") != sha(watchdog_receipt)
        or start.get("stop_preflight_sha256") != sha(stop_preflight)
        or (output / "session_result.json").exists()):
        raise PermissionError("Worker lacks live supervisor provenance")
    return dict(parent_pid=os.getppid(), supervisor_start_sha256=sha(start_path),
        dispatch_sha256=sha(dispatch_path), execution_plan_sha256=execution_plan_sha256,
        prepared_plan_sha256=prepared_plan_sha256)


def check_plan(prepared, expected_sha256):
    if expected_sha256 != PREPARED_PLAN_SHA256:
        raise PermissionError("Prepared input plan SHA-256 differs from reviewed plan")
    plan, rows = frozen.checked_plan(prepared, expected_sha256)
    if plan["synthetic"] is not False or len(rows) != 5:
        raise PermissionError("Live execution requires the exact five-report real package")
    return plan, rows


def remaining_before_stop(grant, current=None):
    current = current or datetime.now(timezone.utc)
    stop_at = resource_gate.parse_time(grant["watchdog_stop_at"])
    return (stop_at - current).total_seconds()


def check_time_budget(grant, started, current=None, minimum_remaining_seconds=0):
    if (time.monotonic() - started >= runtime_config()["maximum_inference_seconds"] or
        remaining_before_stop(grant, current) <= minimum_remaining_seconds):
        raise TimeoutError("Inference reached its fixed limit or watchdog stop reserve")


def bounded_supervisor_timeout(grant, current=None):
    remaining = remaining_before_stop(grant, current) - 1.0  # one-second launch margin
    timeout = min(runtime_config()["maximum_inference_seconds"] - 1.0, remaining)
    if timeout <= 0:
        raise TimeoutError("No inference time remains before watchdog stop")
    return timeout


def verify_reported_hardware(names, visible_count, cuda_version, torch_version, cuda_available, native_bf16, free_gib):
    proposal = read(QWEN_EXECUTION / "configs/resource_proposal.json")
    if names != [proposal["gpu_name"]] or visible_count != 1:
        raise RuntimeError("Exact single approved GPU is required")
    if cuda_version != "12.8" or torch_version != "2.8.0+cu128":
        raise RuntimeError("Pinned CUDA and torch builds are required")
    if not cuda_available or not native_bf16:
        raise RuntimeError("CUDA and native BF16 required")
    if free_gib < runtime_config()["minimum_free_gpu_gib"]:
        raise RuntimeError("At least 36 GiB free GPU memory is required")

def verify_hardware_and_versions():
    query = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=15, check=True)
    names = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    driver_query = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=15, check=True)
    driver_versions = [line.strip() for line in driver_query.stdout.splitlines() if line.strip()]
    if len(driver_versions) != 1:
        raise RuntimeError("Could not record the single GPU driver version")
    from v2_runtime import check_versions
    versions = check_versions()
    if versions != runtime_config()["required_versions"]:
        raise RuntimeError("Dependency pins differ from the execution plan")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    free_gib = torch.cuda.mem_get_info(0)[0] / 2**30
    verify_reported_hardware(names, torch.cuda.device_count(), torch.version.cuda, torch.__version__,
        torch.cuda.is_available(), torch.cuda.is_bf16_supported(including_emulation=False), free_gib)
    return dict(gpu_names=names, gpu_driver_version=driver_versions[0],
                cuda_version=torch.version.cuda, torch_version=torch.__version__,
                package_versions=versions, native_bf16=True)


def create_encoder(cache):
    # The frozen helper selects the exact Qwen revision, pinned template, and frozen model recipe.
    return frozen.real_encoder(cache)


def verify_prepared_tokens(prepared, arm, rows, encoder):
    pinned = read(Path(prepared) / "tokenized.json")[arm]
    entries = [encoder.encode(frozen.make_prompt(arm, row["Report"])) for row in rows]
    actual = [dict(rendered_prompt=e.rendered_prompt, input_ids=e.input_ids, input_tokens=e.input_tokens)
              for e in entries]
    if actual != pinned:
        raise ValueError("Input tokenization differs byte-for-byte from prepared artifacts")
    cfg = runtime_config()
    if any(e.input_tokens > cfg["maximum_input_tokens"] or
           e.input_tokens + cfg["maximum_new_tokens"] > cfg["context_limit"] for e in entries):
        raise ValueError("Input or context token cap exceeded")
    return entries


def create_backend(encoder, cache):
    from v2_runtime import HFBackend
    backend = HFBackend(encoder, cache)
    if backend.metadata["gpu_name"] != read(QWEN_EXECUTION / "configs/resource_proposal.json")["gpu_name"]:
        raise RuntimeError("Loaded GPU differs from approved hardware")
    if not backend.torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Native BF16 required")
    if any(p.device.type != "cuda" or
           (p.is_floating_point() and p.dtype != backend.torch.bfloat16)
           for p in backend.model.parameters()):
        raise RuntimeError("Model parameters must remain CUDA BF16")
    if getattr(backend.model.config, "_attn_implementation", None) != "sdpa":
        raise RuntimeError("SDPA is required")
    if backend.eos != QWEN_STOP_IDS:
        raise RuntimeError("Qwen stop IDs differ from [151645, 151643]")
    if backend.context_limit != runtime_config()["context_limit"]:
        raise RuntimeError("Loaded model context limit differs from approved configuration")
    return backend


def independent_decode(tokenizer, result):
    return resource_gate.independent_decode(tokenizer, result["output_token_ids"],
        result["raw_output"], result["decoded_with_special_tokens"])


def validate_generation(result, backend, encoded, report):
    if result.get("generation_status") != "completed":
        raise RuntimeError("Incomplete generation; experiment stops with no retry")
    ids = result.get("output_token_ids", [])
    elapsed = result.get("runtime_seconds")
    if (type(result.get("output_tokens")) is not int or result["output_tokens"] != len(ids)
        or type(result.get("input_tokens")) is not int or result["input_tokens"] != encoded.input_tokens
        or not 0 < len(ids) <= 2048 or type(elapsed) not in (int, float)
        or not math.isfinite(elapsed) or elapsed < 0):
        raise ValueError("Generation token/time counts violate the frozen caps")
    from v2_runtime import generation_status
    if generation_status(ids, QWEN_STOP_IDS, elapsed, 300, 2048) != "completed":
        raise RuntimeError("Saved tokens indicate an incomplete generation; no retry")
    independent_decode(backend.encoder.tokenizer, result)
    primary = frozen.validate_primary(result["raw_output"], report, "completed")
    secondary = frozen.validate_secondary(result["raw_output"], report, "completed")
    return primary, secondary

def durable_generation(backend, encoded, row, run_key, index, attempt_number, pod_id,
                       attempt_stream, raw_stream, parsed_stream):
    """Persist attempt, then raw tokens, then independently verify and parse exactly once."""
    append_durable(attempt_stream, dict(global_attempt=attempt_number, run_key=run_key,
        case_index=index, pod_id=pod_id, started_at=datetime.now(timezone.utc).isoformat(),
        input_sha256=row["report_sha256"]))
    generated_at = time.monotonic()
    value = backend.generate(encoded)
    record = dict(global_attempt=attempt_number, run_key=run_key, case_index=index,
        StudyInstanceUID=row["StudyInstanceUID"], report_sha256=row["report_sha256"],
        rendered_prompt=encoded.rendered_prompt, input_ids=encoded.input_ids,
        input_tokens=encoded.input_tokens, generation=value,
        generation_elapsed_seconds=time.monotonic() - generated_at)
    append_durable(raw_stream, record)
    primary, secondary = validate_generation(value, backend, encoded, row["Report"])
    parsed_record = dict(global_attempt=attempt_number, case_index=index,
        raw_record_sha256=hashlib.sha256(json.dumps(record, sort_keys=True,
            ensure_ascii=False, allow_nan=False).encode()).hexdigest(),
        primary=primary, secondary=secondary)
    append_durable(parsed_stream, parsed_record)
    return value


def run_session(prepared, prepared_plan_sha256, execution_plan_path, execution_plan_sha256,
                authorization, watchdog_receipt, stop_preflight, cache, output,
                supervisor_context, worker_fd, current=None):
    """Run ABBA20 only as a parent-attested child under the process-group supervisor."""
    validate_execution_plan(execution_plan_path, execution_plan_sha256)
    plan, rows = check_plan(prepared, prepared_plan_sha256)
    grant = live_gate(execution_plan_path, execution_plan_sha256, authorization,
                      watchdog_receipt, stop_preflight, current=current)
    output = Path(output).resolve()
    if not output.is_dir() or not supervisor_context:
        raise PermissionError("Supervised parent must create the new private output directory")
    if output.is_relative_to(REPO) and not output.is_relative_to(REPO / "state"):
        raise ValueError("Execution outputs must be under ignored state/ or external private storage")
    worker_context = verify_worker_attestation(worker_fd, output, execution_plan_sha256,
        prepared_plan_sha256, authorization, watchdog_receipt, stop_preflight)
    started = time.monotonic()
    started_at = datetime.now(timezone.utc)
    result = dict(status="failed", prepared_plan_sha256=prepared_plan_sha256,
        execution_plan_sha256=execution_plan_sha256, proposal_sha256=PROPOSAL_SHA256,
        authorization_sha256=sha(authorization), watchdog_receipt_sha256=sha(watchdog_receipt),
        stop_preflight_sha256=sha(stop_preflight), pod_id=grant["pod_id"], run_order=[],
        generation_attempts=0, generations=0, training_steps=0, retries=0, replacement_hosts=0,
        started_at=started_at.isoformat(), supervisor_context=worker_context)
    write_new(output / "session_start.json", result.copy())
    completed = False
    try:
        # Re-check the live gate before touching cache or loading tokenizer/model.
        grant = live_gate(execution_plan_path, execution_plan_sha256, authorization,
                          watchdog_receipt, stop_preflight)
        hw = verify_hardware_and_versions()
        result["hardware_preflight"] = hw
        check_time_budget(grant, started)
        from load_preflight import audit_cache
        audit = audit_cache(cache, "qwen")
        write_new(output / "cache_audit.json", audit)
        if not audit["complete"] or audit["revision"] != execution_plan()["model_revision"]:
            raise RuntimeError("Full pinned Qwen cache verification failed; no downloads permitted")
        encoder = create_encoder(cache)
        arms = {"control": verify_prepared_tokens(prepared, "control", rows, encoder),
                "candidate": verify_prepared_tokens(prepared, "candidate", rows, encoder)}
        write_new(output / "tokenizer_preflight.json", dict(
            model_id=execution_plan()["model_id"], revision=execution_plan()["model_revision"],
            input_tokens={arm: [entry.input_tokens for entry in entries] for arm, entries in arms.items()},
            exact_prepared_parity=True, maximum_input_tokens=8192, context_limit=40960))
        check_time_budget(grant, started)
        # Revalidate authorization/watchdog immediately before model weights are touched.
        grant = live_gate(execution_plan_path, execution_plan_sha256, authorization,
                          watchdog_receipt, stop_preflight)
        backend = create_backend(encoder, cache)
        write_new(output / "model_preflight.json", dict(hardware=hw, backend=backend.metadata,
            eos_ids=backend.eos, attention="sdpa", model_revision=execution_plan()["model_revision"]))
        run_results = []
        generation_index = 0
        for run_key in execution_plan()["run_order"]:
            check_time_budget(grant, started)
            arm = run_key.split("-")[0]
            run_dir = output / run_key
            run_dir.mkdir(mode=0o700)
            raw_path, parsed_path, attempt_path = (run_dir / "raw.jsonl", run_dir / "parsed.jsonl", run_dir / "attempts.jsonl")
            with raw_path.open("x") as raw_stream, parsed_path.open("x") as parsed_stream, attempt_path.open("x") as attempt_stream:
                for stream in (raw_stream, parsed_stream, attempt_stream):
                    os.chmod(stream.name, 0o600)
                for index, (row, encoded) in enumerate(zip(rows, arms[arm])):
                    # Each call rechecks watchdog process identity and the hard time reserve.
                    if generation_index >= 20:
                        raise RuntimeError("Maximum of 20 generations reached")
                    pre_generation_gate(watchdog_receipt, grant, stop_preflight, started)
                    generation_index += 1
                    result["generation_attempts"] = generation_index
                    durable_generation(backend, encoded, row, run_key, index, generation_index,
                        grant["pod_id"], attempt_stream, raw_stream, parsed_stream)
                    result["generations"] = generation_index
                    check_time_budget(grant, started)
                raw_stream.close(); parsed_stream.close(); attempt_stream.close()
            run_results.append(dict(run_key=run_key, status="completed", generations=5))
            result["run_order"].append(run_key)
        completed = generation_index == 20 and result["run_order"] == execution_plan()["run_order"]
        result.update(status="completed" if completed else "failed", generations=generation_index,
                      run_results=run_results, finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        result.update(status="failed", failure_type=type(error).__name__,
                      finished_at=datetime.now(timezone.utc).isoformat())
        if not isinstance(error, Exception):
            raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        result["generation_attempts"] = sum(len((p).read_bytes().splitlines()) for p in output.rglob("attempts.jsonl"))
        result["generations"] = sum(len((p).read_bytes().splitlines()) for p in output.rglob("raw.jsonl"))
        result["artifacts"] = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*"))
                                if p.is_file() and p.name != "session_result.json"}
        write_new(output / "session_result.json", result)
    return completed


def write_supervisor_receipt(output, receipt):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_new(output / "supervisor_result.json", receipt)


def launch_supervised(prepared, prepared_plan_sha256, execution_plan_path, execution_plan_sha256,
                      authorization, watchdog_receipt, stop_preflight, cache, output):
    """Launch the whole inference session in a killable child process group."""
    validate_execution_plan(execution_plan_path, execution_plan_sha256)
    check_plan(prepared, prepared_plan_sha256)
    grant = live_gate(execution_plan_path, execution_plan_sha256, authorization,
                      watchdog_receipt, stop_preflight)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite or resume an execution directory")
    if output.is_relative_to(REPO) and not output.is_relative_to(REPO / "state"):
        raise ValueError("Execution outputs must be under ignored state/ or external private storage")
    timeout = bounded_supervisor_timeout(grant)
    output.mkdir(parents=True, mode=0o700)
    nonce = secrets.token_hex(32)
    start = dict(parent_pid=os.getpid(), prepared_plan_sha256=prepared_plan_sha256,
        execution_plan_sha256=execution_plan_sha256, authorization_sha256=sha(authorization),
        watchdog_receipt_sha256=sha(watchdog_receipt), stop_preflight_sha256=sha(stop_preflight),
        provider_deadline=grant["provider_deadline"], watchdog_stop_at=grant["watchdog_stop_at"],
        maximum_supervisor_timeout_seconds=timeout, started_at=datetime.now(timezone.utc).isoformat())
    write_new(output / "supervisor_start.json", start)
    write_new(output / "supervisor_dispatch.json", dict(
        supervisor_start_sha256=sha(output / "supervisor_start.json"), nonce_sha256=digest(nonce)))
    reader, writer = os.pipe()
    command = [sys.executable, str(Path(__file__).resolve()), "--execute", "--_worker",
        "--prepared", str(Path(prepared).resolve()), "--prepared-plan-sha256", prepared_plan_sha256,
        "--execution-plan", str(Path(execution_plan_path).resolve()),
        "--execution-plan-sha256", execution_plan_sha256,
        "--authorization", str(Path(authorization).resolve()),
        "--watchdog-receipt", str(Path(watchdog_receipt).resolve()),
        "--stop-preflight", str(Path(stop_preflight).resolve()),
        "--cache", str(Path(cache).resolve()), "--output", str(output),
        "--worker-fd", str(reader)]
    started = time.monotonic()
    try:
        with os.fdopen(writer, "w") as stream:
            json.dump(dict(nonce=nonce, parent_pid=os.getpid(),
                execution_plan_sha256=execution_plan_sha256,
                prepared_plan_sha256=prepared_plan_sha256), stream)
        # Recompute after local setup so file/pipe work cannot consume the stop reserve.
        timeout = bounded_supervisor_timeout(grant)
        ok, process_status = supervise(command, timeout, pass_fds=(reader,))
        ended = time.monotonic()
    except BaseException as error:
        try:
            os.close(writer)
        except OSError:
            pass
        failure = dict(status="failed", process_status=type(error).__name__,
            hard_timeout_seconds=timeout, elapsed_seconds=time.monotonic()-started,
            prepared_plan_sha256=prepared_plan_sha256, execution_plan_sha256=execution_plan_sha256,
            proposal_sha256=PROPOSAL_SHA256, pod_id=grant["pod_id"],
            partial_artifacts_sha256={str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*"))
                                      if p.is_file() and p.name != "supervisor_result.json"})
        try:
            write_supervisor_receipt(output, failure)
        except Exception:
            pass
        raise
    finally:
        os.close(reader)
    result_path = output / "session_result.json"
    child_result = None
    if result_path.exists():
        try:
            child_result = read(result_path)
        except Exception:
            process_status = "invalid_child_result"
    child_complete = bool(child_result and child_result.get("status") == "completed"
        and child_result.get("execution_plan_sha256") == execution_plan_sha256
        and child_result.get("prepared_plan_sha256") == prepared_plan_sha256
        and child_result.get("generations") == 20
        and child_result.get("run_order") == execution_plan()["run_order"])
    receipt = dict(status="completed" if ok and child_complete else "failed",
        process_status=process_status, hard_timeout_seconds=timeout,
        elapsed_seconds=ended-started, prepared_plan_sha256=prepared_plan_sha256,
        execution_plan_sha256=execution_plan_sha256, proposal_sha256=PROPOSAL_SHA256,
        authorization_sha256=sha(authorization), watchdog_receipt_sha256=sha(watchdog_receipt),
        stop_preflight_sha256=sha(stop_preflight), pod_id=grant["pod_id"],
        child_result_sha256=sha(result_path) if result_path.exists() else None,
        child_status=child_result.get("status") if child_result else None,
        partial_artifacts_sha256={str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*"))
                                  if p.is_file() and p.name != "supervisor_result.json"},
        provider_stop_verified=False, billing_stop_verified=False)
    write_supervisor_receipt(output, receipt)
    return receipt["status"] == "completed"


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepared-plan-sha256", required=True)
    parser.add_argument("--execution-plan", type=Path, default=ROOT / "configs/execution_plan.json")
    parser.add_argument("--execution-plan-sha256", required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--watchdog-receipt", type=Path, required=True)
    parser.add_argument("--stop-preflight", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Required in addition to a deliberate config enablement")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.execute or not runtime_config()["execution_enabled"]:
        raise SystemExit("Real execution is disabled by default; no authorization or model access attempted")
    if args._worker:
        context = run_session(args.prepared, args.prepared_plan_sha256, args.execution_plan,
            args.execution_plan_sha256, args.authorization, args.watchdog_receipt,
            args.stop_preflight, args.cache, args.output, supervisor_context=True,
            worker_fd=args.worker_fd)
        raise SystemExit(0 if context else 1)
    ok = launch_supervised(args.prepared, args.prepared_plan_sha256, args.execution_plan,
        args.execution_plan_sha256, args.authorization, args.watchdog_receipt,
        args.stop_preflight, args.cache, args.output)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
