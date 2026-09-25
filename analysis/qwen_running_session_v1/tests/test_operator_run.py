from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run = load("qwen_operator_test", ROOT / "scripts/qwen_operator_run.py")
guard = load("qwen_stop_guard_test", ROOT / "scripts/external_stop_guard.py")
T0 = datetime(2026, 9, 25, 4, 50, tzinfo=timezone.utc)
NOW = T0 + timedelta(minutes=10)


def session():
    return {
        "schema_version": 1, "pod_name": "low_brown_viper", "pod_id": "rqnenq3mpu0i2g", "system_ram_gb": 50,
        "provider_state": "RUNNING", "observed_at": NOW.isoformat(), "cloud_type": "SECURE",
        "gpu_name": "NVIDIA RTX A6000", "gpu_count": 1, "gpu_vram_gb": 48,
        "region": "EU-SE-1", "container_disk_gb": 80, "persistent_volume_gb": 0,
        "network_volume_id": None, "compute_usd_per_hour": .53, "storage_usd_per_hour": .011,
        "maximum_usd": 3.0, "t0": T0.isoformat(),
        "hard_deadline": (T0 + timedelta(hours=3)).isoformat(),
        "shutdown_at": (T0 + timedelta(minutes=175)).isoformat(),
        "execution_plan_sha256": "a" * 64,
        "prepared_plan_sha256": run.PREPARED_SHA, "model_revision": run.REVISION,
        "prior_session_cost_usd": .41,
    }


def test_exact_active_pod_session_and_deadlines_pass():
    value = run.validate_session(session(), now=NOW)
    assert value["deadline"] == T0 + timedelta(hours=3)
    assert value["shutdown_at"] == T0 + timedelta(minutes=175)
    assert value["inference_cutoff"] == T0 + timedelta(minutes=165)
    assert run.require_time_for_run(session(), now=T0 + timedelta(minutes=20)) == 1800


@pytest.mark.parametrize("key,value", [
    ("pod_id", "old-pod-id"), ("pod_name", "rsna-knee-rtx6000ada-smoke"),
    ("provider_state", "STOPPED"), ("gpu_name", "NVIDIA RTX 6000 Ada Generation"),
    ("gpu_count", 2), ("gpu_vram_gb", 47), ("region", "US-WA-1"),
    ("cloud_type", "COMMUNITY"), ("container_disk_gb", 81),
    ("persistent_volume_gb", 1), ("network_volume_id", "volume-1"),
    ("prepared_plan_sha256", "0" * 64), ("model_revision", "latest"),
])
def test_identity_or_scope_drift_fails(key, value):
    data = session(); data[key] = value
    with pytest.raises(run.SessionError):
        run.validate_session(data, now=NOW)


@pytest.mark.parametrize("key,value", [
    ("compute_usd_per_hour", .841), ("storage_usd_per_hour", .013),
    ("maximum_usd", 3.01), ("prior_session_cost_usd", -1),
])
def test_rate_budget_and_prior_cost_are_checked(key, value):
    data = session(); data[key] = value
    with pytest.raises(run.SessionError):
        run.validate_session(data, now=NOW)


def test_deadline_cannot_reset_or_extend_and_old_session_is_rejected():
    data = session(); data["hard_deadline"] = (T0 + timedelta(hours=4)).isoformat()
    with pytest.raises(run.SessionError, match=r"T0\+180"):
        run.validate_session(data, now=NOW)
    with pytest.raises(run.SessionError, match="deadline"):
        run.validate_session(session(), now=T0 + timedelta(hours=3))


def test_inference_requires_full_hard_cap_plus_fifteen_minute_copy_reserve():
    data = session(); now = T0 + timedelta(minutes=136)
    data["observed_at"] = now.isoformat()
    with pytest.raises(TimeoutError):
        run.require_time_for_run(data, now=now)


def test_plan_is_not_the_old_stopped_state_plan_and_hardware_gate_has_a6000():
    policy = json.loads((ROOT / "configs/session_policy.json").read_text())
    assert policy["maximum_provider_seconds"] == 10800
    assert policy["maximum_session_usd"] == 3.0
    assert policy["shutdown_at_seconds"] == 10500
    assert run.EXPECTED_GPU == "NVIDIA RTX A6000"
    assert run.EXPECTED_POD_ID_SHA256 == __import__("hashlib").sha256(b"rqnenq3mpu0i2g").hexdigest()


def test_actual_a6000_mib_capacity_and_free_memory(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(run, "qwen_runtime", lambda: None)
    monkeypatch.setattr(run, "observe_hardware", lambda runtime: {"gpu_names": [run.EXPECTED_GPU]})
    responses = iter([SimpleNamespace(stdout=""), SimpleNamespace(stdout="46068, 45489\n")])
    monkeypatch.setattr(run.subprocess, "run", lambda *a, **kw: next(responses))
    assert run.verify_hardware(session())["total_gpu_mib"] == 46068
    responses = iter([SimpleNamespace(stdout=""), SimpleNamespace(stdout="46068, 35000\n")])
    with pytest.raises(run.SessionError):
        run.verify_hardware(session())
