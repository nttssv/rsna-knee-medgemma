import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qwen_live_alt_test", ROOT / "scripts/qwen_live.py")
live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live)
PROPOSAL = live.read(live.ALT_RESOURCE / "configs/resource_proposal.json")
GPU_NAMES = [item["provider_name"] for item in PROPOSAL["gpu_allowlist"]]
NOW = datetime(2026, 9, 24, 14, 10, tzinfo=timezone.utc)


def grant_for(gpu_name):
    entry = live.alternate_gate.gpu_entry(PROPOSAL, gpu_name)
    return dict(
        approved=True, approval_reference="SYNTHETIC TEST ONLY",
        execution_plan_sha256="f" * 64,
        prepared_plan_sha256=live.PREPARED_PLAN_SHA256,
        resource_proposal_sha256=live.PROPOSAL_SHA256,
        pod_id="synthetic-pod", gpu_name=gpu_name, gpu_count=1, cloud_type="SECURE",
        region="US-WA-1", container_disk_gb=80, persistent_volume_gb=0,
        network_volume_id=None, actual_compute_usd_per_hour=entry["secure_compute_usd_per_hour_ceiling"],
        actual_storage_usd_per_hour=0.012, maximum_usd=1.5,
        approved_at="2026-09-24T14:00:00+00:00",
        provider_start_requested_at="2026-09-24T14:05:00+00:00",
        provider_deadline="2026-09-24T15:05:00+00:00",
        watchdog_stop_at="2026-09-24T15:00:00+00:00")


def write_private(path, value):
    path.write_text(json.dumps(value))
    path.chmod(0o600)
    return path


@pytest.mark.parametrize("gpu_name", GPU_NAMES)
def test_every_reviewed_gpu_passes_exact_grant_and_hardware(gpu_name, tmp_path):
    grant = grant_for(gpu_name)
    path = write_private(tmp_path / "grant.json", grant)
    checked = live.validate_user_approval(path, "f" * 64, current=NOW)
    assert checked["gpu_name"] == gpu_name
    live.verify_reported_hardware([gpu_name], 1, "12.8", "2.8.0+cu128", True, True, 36.0, gpu_name)


@pytest.mark.parametrize("gpu_name", ["NVIDIA H100 80GB HBM3", "NVIDIA A100-SXM4-80GB", "RTX A5000"])
def test_unlisted_gpu_fails(gpu_name):
    with pytest.raises(PermissionError, match="allowlist"):
        live.alternate_gate.validate_selected_gpu(PROPOSAL, gpu_name, 1)


@pytest.mark.parametrize("approved_gpu", GPU_NAMES)
def test_different_but_allowlisted_visible_gpu_fails(approved_gpu):
    alternate = next(name for name in GPU_NAMES if name != approved_gpu)
    with pytest.raises(RuntimeError, match="specifically approved"):
        live.verify_reported_hardware([alternate], 1, "12.8", "2.8.0+cu128", True, True, 48.0, approved_gpu)


def test_gpu_count_drift_fails():
    with pytest.raises(PermissionError, match="Exactly one"):
        live.alternate_gate.validate_selected_gpu(PROPOSAL, GPU_NAMES[0], 2)
    with pytest.raises(RuntimeError, match="Exactly one"):
        live.verify_reported_hardware([GPU_NAMES[0], GPU_NAMES[0]], 2, "12.8", "2.8.0+cu128", True, True, 48, GPU_NAMES[0])


@pytest.mark.parametrize("field,value", [
    ("cloud_type", "COMMUNITY"), ("container_disk_gb", 81),
    ("persistent_volume_gb", 1), ("network_volume_id", "netvol-other"), ("gpu_count", 2),
])
def test_exact_resource_drift_fails(tmp_path, field, value):
    grant = grant_for(GPU_NAMES[0])
    grant[field] = value
    path = write_private(tmp_path / "grant.json", grant)
    with pytest.raises(PermissionError):
        live.validate_user_approval(path, "f" * 64, current=NOW)


@pytest.mark.parametrize("gpu_name", GPU_NAMES)
def test_per_gpu_rate_ceiling_is_enforced(gpu_name, tmp_path):
    grant = grant_for(gpu_name)
    grant["actual_compute_usd_per_hour"] += 0.001
    path = write_private(tmp_path / "grant.json", grant)
    with pytest.raises(PermissionError, match="ceiling"):
        live.validate_user_approval(path, "f" * 64, current=NOW)


def test_provider_observation_must_match_approved_pod_gpu_and_region():
    grant = grant_for(GPU_NAMES[0])
    observed = {key: grant[key] for key in live.alternate_gate.OBSERVED_FIELDS}
    live.alternate_gate.verify_observed_resource(grant, observed)
    observed["region"] = "US-CA-1"
    with pytest.raises(PermissionError, match="region"):
        live.alternate_gate.verify_observed_resource(grant, observed)


def test_new_plan_hash_is_required_and_base_plan_is_not_the_grant_binding(tmp_path):
    grant = grant_for(GPU_NAMES[0])
    path = write_private(tmp_path / "grant.json", grant)
    with pytest.raises(PermissionError, match="exact new execution plan"):
        live.validate_user_approval(path, "e" * 64, current=NOW)
    assert grant["execution_plan_sha256"] != live.BASE_EXECUTION_PLAN_SHA256


def test_live_gate_stays_disabled_before_opening_private_inputs(tmp_path):
    with pytest.raises(PermissionError, match="disabled"):
        live.live_gate(tmp_path / "plan", "0" * 64, tmp_path / "grant", tmp_path / "watchdog",
                       tmp_path / "preflight", tmp_path / "observation")


def test_execution_default_and_fixed_scientific_scope():
    assert live.runtime_config()["execution_enabled"] is False
    plan = live.execution_plan()
    assert plan["prepared_plan_sha256"] == live.PREPARED_PLAN_SHA256
    assert plan["prior_execution_plan_sha256"] == live.BASE_EXECUTION_PLAN_SHA256
    assert plan["resource_proposal_sha256"] == live.PROPOSAL_SHA256
    assert set(plan["gpu_allowlist"]) == set(GPU_NAMES)
    assert plan["run_order"] == ["control-1", "candidate-1", "candidate-2", "control-2"]
    assert plan["maximum_generations"] == 20 and plan["reports_per_run"] == 5
    assert plan["precision"] == "bfloat16" and plan["attention_implementation"] == "sdpa"
    assert plan["batch_size"] == 1
    assert all(plan[name] is False for name in ("training", "validation_inference",
        "full_development_inference", "bulk_extraction", "retry_or_repair",
        "real_execution_enabled_by_default"))


def test_execution_plan_hash_binds_all_runtime_and_resource_sources():
    plan_path = ROOT / "configs/execution_plan.json"
    plan_sha = live.sha(plan_path)
    plan = live.validate_execution_plan(plan_path, plan_sha)
    assert plan["runtime_source_sha256"] == live.sha(ROOT / "scripts/qwen_live.py")
    assert plan["runtime_policy_sha256"] == live.sha(ROOT / "configs/runtime.json")
    assert plan["resource_gate_sha256"] == live.sha(live.ALT_RESOURCE / "scripts/resource_alt_gate.py")
    assert plan["resource_proposal_sha256"] == live.sha(live.ALT_RESOURCE / "configs/resource_proposal.json")


def test_supervisor_watchdog_and_decode_protections_are_retained():
    source = (ROOT / "scripts/qwen_live.py").read_text()
    assert "verify_live_watchdog(watchdog_receipt, grant, stop_preflight)" in source
    assert "generation_index >= 20" in source
    assert "supervise(command, timeout, pass_fds=(reader,))" in source
    assert "independent_decode(backend.encoder.tokenizer, result)" in source
    assert "write_new(output / \"session_result.json\", result)" in source
    assert "partial_artifacts_sha256" in source
    assert "automatic_retries\": false" in (ROOT / "configs/runtime.json").read_text()


def test_provider_observation_exact_schema_is_required():
    grant = grant_for(GPU_NAMES[0])
    observation = {key: grant[key] for key in live.alternate_gate.OBSERVED_FIELDS}
    live.alternate_gate.verify_observed_resource(grant, observation)
    observation["unexpected"] = True
    with pytest.raises(PermissionError, match="missing or unexpected"):
        live.alternate_gate.verify_observed_resource(grant, observation)
