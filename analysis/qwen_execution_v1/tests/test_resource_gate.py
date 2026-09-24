"""CPU-only resource-gate tests; no provider calls or model loading."""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import resource_gate as gate
import stop_watchdog as watch


@pytest.fixture
def authorization(tmp_path):
    current = datetime(2026, 9, 17, 10, tzinfo=timezone.utc)
    grant = {
        "approved": True,
        "approval_reference": "SYNTHETIC TEST ONLY",
        "plan_sha256": "f" * 64,
        "proposal_sha256": gate.sha(gate.ROOT / "configs/resource_proposal.json"),
        "approved_at": (current - timedelta(minutes=6)).isoformat(),
        "provider_start_requested_at": (current - timedelta(minutes=5)).isoformat(),
        "provider_deadline": (current + timedelta(minutes=50)).isoformat(),
        "maximum_usd": 1.5,
        "actual_compute_usd_per_hour": 0.84,
        "actual_storage_usd_per_hour": 0.023,
        "pod_id": "synthetic-pod",
        "gpu_name": "NVIDIA RTX 6000 Ada Generation",
        "gpu_count": 1
    }
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(grant))
    path.chmod(0o600)
    return path, current


def test_valid_fixture_authorization(authorization):
    path, current = authorization
    assert gate.validate_authorization(path, "f" * 64, current)["approved"] is True


@pytest.mark.parametrize("field,value", [
    ("approved", False), ("approval_reference", ""), ("plan_sha256", "wrong"),
    ("proposal_sha256", "wrong"), ("gpu_name", "NVIDIA A100"), ("gpu_count", 2),
    ("gpu_count", True), ("pod_id", ""), ("maximum_usd", 1.51),
    ("actual_compute_usd_per_hour", 0.85), ("actual_storage_usd_per_hour", 0.024),
    ("actual_compute_usd_per_hour", -1),
    ("provider_deadline", "2026-09-17T12:00:00+00:00"),
    ("provider_deadline", "2026-09-17T10:04:00+00:00"),
    ("provider_deadline", "2026-09-17T11:00:00")
])
def test_bad_authorization_rejected(authorization, field, value):
    path, current = authorization
    record = json.loads(path.read_text())
    record[field] = value
    path.write_text(json.dumps(record))
    with pytest.raises((PermissionError, ValueError)):
        gate.validate_authorization(path, "f" * 64, current)


def test_world_readable_authorization_rejected(authorization):
    path, current = authorization
    path.chmod(0o644)
    with pytest.raises(PermissionError):
        gate.validate_authorization(path, "f" * 64, current)


def test_world_readable_watchdog_receipt_rejected(authorization, tmp_path):
    path, current = authorization
    grant = json.loads(path.read_text())
    receipt = tmp_path / "watchdog.json"
    receipt.write_text(json.dumps({"status": "armed"}))
    receipt.chmod(0o644)
    with pytest.raises(PermissionError, match="private"):
        gate.verify_watchdog(receipt, grant, gate.ROOT / "scripts/stop_watchdog.py", tmp_path / "unused", current)


def test_watchdog_environment_must_match_approved_pod(authorization, tmp_path, monkeypatch):
    path, current = authorization
    grant = json.loads(path.read_text())
    script = gate.ROOT / "scripts/stop_watchdog.py"
    preflight = tmp_path / "control-preflight.json"
    control = {
        "status": "verified", "pod_id": grant["pod_id"], "cli_style": "modern",
        "state_before": "EXITED", "stop_command": ["/usr/local/bin/runpodctl", "pod", "stop", grant["pod_id"]],
        "stop_exit_code": 0, "state_after": "EXITED",
        "verified_at": (gate.parse_time(grant["provider_start_requested_at"]) - timedelta(minutes=1)).isoformat(),
        "same_credential_context": True
    }
    preflight.write_text(json.dumps(control))
    preflight.chmod(0o600)
    receipt = tmp_path / "watchdog.json"
    receipt.write_text(json.dumps({
        "status": "armed", "pod_id": grant["pod_id"], "deadline": grant["provider_deadline"],
        "script_sha256": gate.sha(script), "cli_syntax_and_read_access_verified": True,
        "stop_access_preflight_verified": True, "stop_access_preflight_pod_id": grant["pod_id"],
        "stop_preflight_sha256": gate.sha(preflight),
        "provider_stop_verified": False, "armed_at": current.isoformat(), "pid": os.getpid(),
        "command": ["/usr/local/bin/runpodctl", "pod", "stop", grant["pod_id"]]
    }))
    receipt.chmod(0o600)
    monkeypatch.setenv("RUNPOD_POD_ID", "another-pod")
    with pytest.raises(PermissionError, match="mismatched"):
        gate.verify_watchdog(receipt, grant, script, preflight, current)


def test_quote_expiry_rejected(authorization):
    path, _ = authorization
    with pytest.raises(PermissionError, match="expired"):
        gate.validate_authorization(path, "f" * 64, datetime(2026, 9, 26, tzinfo=timezone.utc))


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        raw = '{"ACL":{"label":"negative"}}'
        return raw + "<|im_end|>" if ids and ids[-1] == 151645 else raw


def test_independent_decode_is_exact():
    raw = '{"ACL":{"label":"negative"}}'
    special = raw + "<|im_end|>"
    assert gate.independent_decode(FakeTokenizer(), [1, 2, 151645], raw, special) == {
        "raw_output": raw, "decoded_with_special_tokens": special
    }
    with pytest.raises(ValueError, match="differs"):
        gate.independent_decode(FakeTokenizer(), [1, 2, 151645], raw + " ", special)
    with pytest.raises(ValueError, match="special-token"):
        gate.independent_decode(FakeTokenizer(), [1, 2, 151645], raw, raw)
    with pytest.raises(ValueError):
        gate.independent_decode(FakeTokenizer(), [], raw, raw)


@pytest.mark.parametrize("style,expected", [("modern", ["pod", "stop"]), ("legacy", ["stop", "pod"])])
def test_watchdog_targets_only_its_own_pod(style, expected, monkeypatch):
    monkeypatch.setenv("RUNPOD_POD_ID", "synthetic-pod")
    assert watch.command("synthetic-pod", style, "runpodctl") == ["runpodctl"] + expected + ["synthetic-pod"]
    with pytest.raises(PermissionError):
        watch.command("another-pod", style, "runpodctl")


def test_watchdog_stop_attempts_are_bounded():
    calls = []
    current = datetime(2026, 9, 17, 10, tzinfo=timezone.utc)
    def run(command, **kwargs):
        calls.append(command)
        raise OSError("synthetic")
    assert watch.wait_and_stop(current, ["FAKE-STOP"], clock=lambda: current, sleep=lambda _: None, run=run) is False
    assert len(calls) == 3


def test_cli_preflight_does_not_issue_stop(monkeypatch):
    monkeypatch.setenv("RUNPOD_POD_ID", "synthetic-pod")
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="stop usage" if "--help" in command else "synthetic-pod")
    assert watch.verify_cli("synthetic-pod", "modern", "runpodctl", run)
    assert len(calls) == 2 and calls[0][-1] == "--help" and "get" in calls[1]


def test_stop_preflight_receipt_requires_stopped_pod(tmp_path):
    current = datetime(2026, 9, 17, 10, tzinfo=timezone.utc)
    path = tmp_path / "control-preflight.json"
    receipt = {
        "status": "verified", "pod_id": "synthetic-pod", "cli_style": "modern",
        "state_before": "EXITED", "stop_command": ["/usr/local/bin/runpodctl", "pod", "stop", "synthetic-pod"],
        "stop_exit_code": 0, "state_after": "EXITED", "verified_at": current.isoformat(),
        "same_credential_context": True
    }
    path.write_text(json.dumps(receipt))
    path.chmod(0o600)
    assert watch.verify_stop_preflight(path, "synthetic-pod", "modern", "/usr/local/bin/runpodctl", current)["status"] == "verified"
    receipt["state_before"] = "RUNNING"
    path.write_text(json.dumps(receipt))
    with pytest.raises(PermissionError, match="already-stopped"):
        watch.verify_stop_preflight(path, "synthetic-pod", "modern", "/usr/local/bin/runpodctl", current)


def test_stop_preflight_receipt_rejects_another_pod(tmp_path):
    current = datetime(2026, 9, 17, 10, tzinfo=timezone.utc)
    path = tmp_path / "control-preflight.json"
    path.write_text(json.dumps({
        "status": "verified", "pod_id": "another-pod", "cli_style": "modern",
        "state_before": "EXITED", "stop_command": ["/usr/local/bin/runpodctl", "pod", "stop", "another-pod"],
        "stop_exit_code": 0, "state_after": "EXITED", "verified_at": current.isoformat(),
        "same_credential_context": True
    }))
    path.chmod(0o600)
    with pytest.raises(PermissionError, match="exact pod"):
        watch.verify_stop_preflight(path, "synthetic-pod", "modern", "/usr/local/bin/runpodctl", current)
