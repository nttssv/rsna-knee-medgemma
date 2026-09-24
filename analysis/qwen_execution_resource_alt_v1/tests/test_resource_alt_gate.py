import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import resource_alt_gate


PROPOSAL = json.loads((ROOT / "configs/resource_proposal.json").read_text())
NOW = datetime(2026, 9, 24, 15, 30, tzinfo=timezone.utc)


@pytest.mark.parametrize("entry", PROPOSAL["gpu_allowlist"], ids=lambda x: x["provider_name"])
def test_every_reviewed_gpu_passes_allowlist_gate(entry):
    assert resource_alt_gate.validate_selected_gpu(PROPOSAL, entry["provider_name"], 1) == entry


@pytest.mark.parametrize("gpu_name", [
    "NVIDIA H100 SXM",
    "NVIDIA RTX 4090",
    "NVIDIA A100 PCIe",
    "RTX 6000 Ada",
    "",
])
def test_any_gpu_outside_exact_allowlist_fails(gpu_name):
    with pytest.raises(PermissionError, match="not in the versioned resource allowlist"):
        resource_alt_gate.validate_selected_gpu(PROPOSAL, gpu_name, 1)


@pytest.mark.parametrize("count", [0, 2, True, "1"])
def test_gpu_count_must_be_exactly_one(count):
    with pytest.raises(PermissionError, match="Exactly one approved GPU"):
        resource_alt_gate.validate_selected_gpu(PROPOSAL, "NVIDIA A40", count)


def grant_for(gpu_name="NVIDIA A40"):
    entry = resource_alt_gate.gpu_entry(PROPOSAL, gpu_name)
    return {
        "approved": True,
        "approval_reference": "synthetic-test-only",
        "execution_plan_sha256": PROPOSAL["base_execution_plan_sha256"],
        "prepared_plan_sha256": PROPOSAL["prepared_plan_sha256"],
        "resource_proposal_sha256": "proposal-hash-fixture",
        "pod_id": "synthetic-pod-fixture",
        "gpu_name": gpu_name,
        "gpu_count": 1,
        "cloud_type": "SECURE",
        "region": "US-WA-1",
        "container_disk_gb": 80,
        "persistent_volume_gb": 0,
        "network_volume_id": None,
        "actual_compute_usd_per_hour": entry["secure_compute_usd_per_hour_ceiling"],
        "actual_storage_usd_per_hour": 0.012,
        "maximum_usd": 1.50,
        "approved_at": "2026-09-24T15:00:00+00:00",
        "provider_start_requested_at": "2026-09-24T15:01:00+00:00",
        "provider_deadline": "2026-09-24T16:01:00+00:00",
        "watchdog_stop_at": "2026-09-24T15:56:00+00:00",
    }


def test_grant_binds_gpu_identity_and_actual_rate():
    grant = grant_for("NVIDIA L40S")
    selected = resource_alt_gate.validate_grant(
        PROPOSAL, grant, "proposal-hash-fixture", lambda value: datetime.fromisoformat(value), current=NOW)
    assert selected["provider_name"] == "NVIDIA L40S"
    observed = {key: grant[key] for key in resource_alt_gate.OBSERVED_FIELDS}
    resource_alt_gate.verify_observed_resource(grant, observed)
    resource_alt_gate.verify_reported_hardware(PROPOSAL, grant, ["NVIDIA L40S"], 1)


@pytest.mark.parametrize(("field", "value"), [
    ("gpu_name", "NVIDIA H100 SXM"),
    ("gpu_count", 2),
    ("cloud_type", "COMMUNITY"),
    ("region", ""),
    ("container_disk_gb", 100),
    ("persistent_volume_gb", 10),
    ("network_volume_id", "some-volume"),
    ("actual_compute_usd_per_hour", 1.10),
    ("actual_storage_usd_per_hour", 0.013),
    ("maximum_usd", 1.51),
])
def test_grant_rejects_changed_gpu_storage_rate_or_budget(field, value):
    grant = grant_for("NVIDIA A40")
    grant[field] = value
    with pytest.raises(PermissionError):
        resource_alt_gate.validate_grant(
            PROPOSAL, grant, "proposal-hash-fixture", lambda value: datetime.fromisoformat(value), current=NOW)


def test_provider_observation_must_match_approved_identity_and_rate():
    grant = grant_for("NVIDIA RTX A6000")
    observed = {key: grant[key] for key in resource_alt_gate.OBSERVED_FIELDS}
    resource_alt_gate.verify_observed_resource(grant, observed)
    observed["actual_compute_usd_per_hour"] -= 0.01
    with pytest.raises(PermissionError, match="actual_compute_usd_per_hour"):
        resource_alt_gate.verify_observed_resource(grant, observed)


def test_expired_proposal_quote_is_rejected():
    proposal = dict(PROPOSAL)
    proposal["quote_valid_until"] = "2026-09-24T15:29:59+00:00"
    grant = grant_for("NVIDIA A40")
    with pytest.raises(PermissionError, match="quote expired"):
        resource_alt_gate.validate_grant(
            proposal, grant, "proposal-hash-fixture", lambda value: datetime.fromisoformat(value), current=NOW)


def test_runtime_rejects_different_reported_gpu():
    grant = grant_for("NVIDIA A40")
    with pytest.raises(RuntimeError, match="differs from the specifically approved"):
        resource_alt_gate.verify_reported_hardware(PROPOSAL, grant, ["NVIDIA L40"], 1)


def test_l40s_still_fits_one_hour_budget():
    price = (1.09 + 0.012) * 1
    assert price == pytest.approx(1.102)
    assert price < PROPOSAL["maximum_approved_usd"]
