"""CPU-verifiable gate for a future Qwen GPU run; never starts or stops a pod."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def parse_time(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("Timezone-aware timestamp required")
    return result


def validate_authorization(path, plan_sha256, current=None):
    """Validate a private, exact-plan grant without performing provider actions."""
    path = Path(path)
    if path.stat().st_mode & 0o077:
        raise PermissionError("Authorization must be private (0600)")
    grant = read(path)
    proposal = read(ROOT / "configs/resource_proposal.json")
    runtime = read(ROOT / "configs/runtime.json")
    required = {
        "approved", "approval_reference", "plan_sha256", "proposal_sha256",
        "approved_at", "provider_start_requested_at", "provider_deadline",
        "maximum_usd", "actual_compute_usd_per_hour",
        "actual_storage_usd_per_hour", "pod_id", "gpu_name", "gpu_count"
    }
    if set(grant) != required or grant["approved"] is not True:
        raise PermissionError("Exact private user approval record required")
    if not str(grant["approval_reference"]).strip():
        raise PermissionError("Approval reference is empty")
    if grant["plan_sha256"] != plan_sha256:
        raise PermissionError("Approval is for another plan")
    if grant["proposal_sha256"] != sha(ROOT / "configs/resource_proposal.json"):
        raise PermissionError("Approval is for another resource proposal")
    if grant["gpu_name"] != proposal["gpu_name"] or type(grant["gpu_count"]) is not int or grant["gpu_count"] != 1:
        raise PermissionError("Approval does not cover this GPU")
    if not str(grant["pod_id"]).strip():
        raise PermissionError("Pod identity is required")
    current = current or datetime.now(timezone.utc)
    if current > parse_time(proposal["quote_valid_until"]):
        raise PermissionError("Resource quote expired")
    approved = parse_time(grant["approved_at"])
    started = parse_time(grant["provider_start_requested_at"])
    deadline = parse_time(grant["provider_deadline"])
    seconds = (deadline - started).total_seconds()
    if not approved <= started <= current < deadline or not 0 < seconds <= runtime["maximum_provider_seconds"]:
        raise PermissionError("Approval timestamps or provider window are invalid")
    values = [grant[k] for k in ("maximum_usd", "actual_compute_usd_per_hour", "actual_storage_usd_per_hour")]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
        raise PermissionError("Invalid cost values")
    if not 0 < grant["maximum_usd"] <= runtime["maximum_approved_usd"]:
        raise PermissionError("Budget exceeds proposal")
    if grant["actual_compute_usd_per_hour"] > proposal["compute_usd_per_hour_ceiling"]:
        raise PermissionError("Compute quote exceeds proposal")
    if grant["actual_storage_usd_per_hour"] > proposal["storage_usd_per_hour_ceiling"]:
        raise PermissionError("Storage quote exceeds proposal")
    if sum(values[1:]) * seconds / 3600 > grant["maximum_usd"]:
        raise PermissionError("Window estimate exceeds approved budget")
    if (deadline - current).total_seconds() <= runtime["reserved_copy_stop_seconds"]:
        raise PermissionError("Copy-and-stop reserve is unavailable")
    return grant


def verify_watchdog(receipt_path, grant, script_path, stop_preflight_path, current=None):
    receipt_path = Path(receipt_path)
    stop_preflight_path = Path(stop_preflight_path)
    if receipt_path.stat().st_mode & 0o077:
        raise PermissionError("Watchdog receipt must be private (0600)")
    if stop_preflight_path.stat().st_mode & 0o077:
        raise PermissionError("Stop preflight receipt must be private (0600)")
    receipt = read(receipt_path)
    preflight = read(stop_preflight_path)
    current = current or datetime.now(timezone.utc)
    deadline = parse_time(grant["provider_deadline"])
    started = parse_time(grant["provider_start_requested_at"])
    armed = parse_time(receipt.get("armed_at", ""))
    command = receipt.get("command")
    valid_command = (isinstance(command, list) and len(command) == 4
        and Path(command[0]).name == "runpodctl" and command[-1] == grant["pod_id"]
        and command[1:] in (["pod", "stop", grant["pod_id"]], ["stop", "pod", grant["pod_id"]]))
    if command and command[1:3] == ["pod", "stop"]:
        cli_style = "modern"
    elif command and command[1:3] == ["stop", "pod"]:
        cli_style = "legacy"
    else:
        cli_style = None
    preflight_time = parse_time(preflight.get("verified_at", ""))
    valid_preflight = (
        valid_command
        and set(preflight) == {"status", "pod_id", "cli_style", "state_before", "stop_command",
            "stop_exit_code", "state_after", "verified_at", "same_credential_context"}
        and preflight["status"] == "verified" and preflight["pod_id"] == grant["pod_id"]
        and preflight["cli_style"] == cli_style and preflight["same_credential_context"] is True
        and preflight["state_before"] in ("EXITED", "STOPPED")
        and preflight["state_after"] in ("EXITED", "STOPPED")
        and preflight["stop_exit_code"] == 0
        and isinstance(preflight["stop_command"], list)
        and len(preflight["stop_command"]) == 4
        and Path(preflight["stop_command"][0]).resolve() == Path(command[0]).resolve()
        and preflight["stop_command"][-3:] == command[1:]
        and preflight_time <= started and (started - preflight_time).total_seconds() <= 86400
    )
    if (receipt.get("status") != "armed" or receipt.get("pod_id") != grant["pod_id"]
            or receipt.get("deadline") != grant["provider_deadline"]
            or receipt.get("script_sha256") != sha(script_path)
            or receipt.get("cli_syntax_and_read_access_verified") is not True
            or receipt.get("stop_access_preflight_verified") is not True
            or receipt.get("stop_access_preflight_pod_id") != grant["pod_id"]
            or receipt.get("stop_preflight_sha256") != sha(stop_preflight_path)
            or not valid_preflight
            or receipt.get("provider_stop_verified") is not False or not valid_command
            or not started <= armed <= current < deadline
            or os.environ.get("RUNPOD_POD_ID") != grant["pod_id"]
            or type(receipt.get("pid")) is not int or receipt["pid"] <= 1):
        raise PermissionError("Live self-stop watchdog is missing or mismatched")
    os.kill(receipt["pid"], 0)
    cmdline = Path("/proc") / str(receipt["pid"]) / "cmdline"
    environ = Path("/proc") / str(receipt["pid"]) / "environ"
    required_args = [str(Path(script_path).resolve()).encode(), b"--pod-id", grant["pod_id"].encode(),
        b"--deadline", grant["provider_deadline"].encode(), b"--confirm-self-stop"]
    if (not cmdline.exists() or not environ.exists()
            or any(arg not in cmdline.read_bytes().split(b"\0") for arg in required_args)
            or b"RUNPOD_POD_ID=" + grant["pod_id"].encode() not in environ.read_bytes().split(b"\0")):
        raise PermissionError("Watchdog process identity is unverified")
    return receipt


def independent_decode(tokenizer, output_token_ids, recorded_raw_output, recorded_decoded_with_special_tokens):
    """Bind saved IDs to both recorded strings, removing one terminal Qwen EOS for raw text."""
    if not output_token_ids or any(type(value) is not int or value < 0 for value in output_token_ids):
        raise ValueError("Invalid saved output token IDs")
    eos_ids = [151645, 151643]
    full = tokenizer.decode(output_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if full.encode("utf-8") != recorded_decoded_with_special_tokens.encode("utf-8"):
        raise ValueError("Independent token decode differs from recorded special-token output")
    body = output_token_ids[:-1] if output_token_ids[-1] in eos_ids else output_token_ids
    decoded = tokenizer.decode(body, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if decoded.encode("utf-8") != recorded_raw_output.encode("utf-8"):
        raise ValueError("Independent token decode differs from recorded raw output")
    return {"raw_output": decoded, "decoded_with_special_tokens": full}
