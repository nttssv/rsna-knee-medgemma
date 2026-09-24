"""CPU-only resource gate for the versioned Qwen GPU allowlist proposal.

This module performs no provider calls and cannot enable or launch execution.
The live execution runtime must call these checks before loading the model.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math


GRANT_FIELDS = {
    "approved",
    "approval_reference",
    "execution_plan_sha256",
    "prepared_plan_sha256",
    "resource_proposal_sha256",
    "pod_id",
    "gpu_name",
    "gpu_count",
    "cloud_type",
    "region",
    "container_disk_gb",
    "persistent_volume_gb",
    "network_volume_id",
    "actual_compute_usd_per_hour",
    "actual_storage_usd_per_hour",
    "maximum_usd",
    "approved_at",
    "provider_start_requested_at",
    "provider_deadline",
    "watchdog_stop_at",
}

OBSERVED_FIELDS = {
    "pod_id",
    "gpu_name",
    "gpu_count",
    "cloud_type",
    "region",
    "container_disk_gb",
    "persistent_volume_gb",
    "network_volume_id",
    "actual_compute_usd_per_hour",
    "actual_storage_usd_per_hour",
}


def gpu_entry(proposal: dict, gpu_name: str) -> dict:
    matches = [item for item in proposal["gpu_allowlist"] if item["provider_name"] == gpu_name]
    if len(matches) != 1:
        raise PermissionError("GPU is not in the versioned resource allowlist")
    return matches[0]


def validate_selected_gpu(proposal: dict, gpu_name: str, gpu_count: int = 1) -> dict:
    """Reject unlisted devices while preserving an exact one-GPU requirement."""
    entry = gpu_entry(proposal, gpu_name)
    if type(gpu_count) is not int or gpu_count != proposal["gpu_count"]:
        raise PermissionError("Exactly one approved GPU is required")
    if entry["vram_gb"] != 48 or not entry["bf16_supported"] or not entry["cuda_12_8_compatible"]:
        raise PermissionError("GPU does not satisfy the reviewed runtime requirements")
    if entry["runtime_recipe_change_required"]:
        raise PermissionError("GPU requires a change to the frozen model recipe")
    return entry


def _finite_nonnegative(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise PermissionError(f"Invalid {label}")
    return float(value)


def validate_grant(proposal: dict, grant: dict, proposal_sha256: str,
                   parse_time, current=None) -> dict:
    """Validate the chosen allowlisted GPU, quote, costs, and fixed time boundary."""
    if set(grant) != GRANT_FIELDS or grant.get("approved") is not True:
        raise PermissionError("Exact private approval record required")
    if not str(grant["approval_reference"]).strip() or not str(grant["pod_id"]).strip():
        raise PermissionError("Approval reference and exact pod ID are required")
    if grant["execution_plan_sha256"] != proposal["base_execution_plan_sha256"]:
        raise PermissionError("Grant is bound to another base execution plan")
    if grant["prepared_plan_sha256"] != proposal["prepared_plan_sha256"]:
        raise PermissionError("Grant is bound to another prepared input plan")
    if grant["resource_proposal_sha256"] != proposal_sha256:
        raise PermissionError("Grant is not bound to this exact resource proposal")

    selected = validate_selected_gpu(proposal, grant["gpu_name"], grant["gpu_count"])
    if grant["cloud_type"] != proposal["cloud_type"]:
        raise PermissionError("Only Secure Cloud is approved")
    if not str(grant["region"]).strip():
        raise PermissionError("Approval must bind one exact provider region")
    if (type(grant["container_disk_gb"]) is not int or grant["container_disk_gb"] != 80
            or type(grant["persistent_volume_gb"]) is not int or grant["persistent_volume_gb"] != 0
            or grant["network_volume_id"] is not None):
        raise PermissionError("Storage differs from the exact proposal")

    compute = _finite_nonnegative(grant["actual_compute_usd_per_hour"], "compute rate")
    storage = _finite_nonnegative(grant["actual_storage_usd_per_hour"], "storage rate")
    maximum = _finite_nonnegative(grant["maximum_usd"], "maximum budget")
    if not 0 < compute <= selected["secure_compute_usd_per_hour_ceiling"]:
        raise PermissionError("Chosen GPU compute rate exceeds its reviewed ceiling")
    if storage > proposal["storage"]["storage_usd_per_hour_ceiling"]:
        raise PermissionError("Storage rate exceeds the reviewed ceiling")
    if not 0 < maximum <= proposal["maximum_approved_usd"]:
        raise PermissionError("Approved budget exceeds the reviewed ceiling")

    now = current or datetime.now(timezone.utc)
    if now > parse_time(proposal["quote_valid_until"]):
        raise PermissionError("Resource proposal quote expired")
    approved = parse_time(grant["approved_at"])
    start = parse_time(grant["provider_start_requested_at"])
    deadline = parse_time(grant["provider_deadline"])
    stop_at = parse_time(grant["watchdog_stop_at"])
    seconds = (deadline - start).total_seconds()
    if (not approved <= start <= now < stop_at < deadline
            or stop_at != deadline - timedelta(seconds=proposal["watchdog_stop_reserve_seconds"])
            or not 0 < seconds <= proposal["provider_window_seconds_ceiling"]):
        raise PermissionError("Approval timestamps or fixed provider/watchdog boundary are invalid")
    estimated = (compute + storage) * seconds / 3600
    if estimated > maximum or now >= stop_at:
        raise PermissionError("Budget exceeded or watchdog reserve already started")
    return selected


def verify_observed_resource(grant: dict, observed: dict) -> None:
    """Require live signed-in provider facts to equal the user-approved grant."""
    if set(observed) != OBSERVED_FIELDS:
        raise PermissionError("Provider observation has missing or unexpected fields")
    for field in OBSERVED_FIELDS:
        if observed[field] != grant[field]:
            raise PermissionError(f"Observed provider resource differs from approval: {field}")


def verify_reported_hardware(proposal: dict, grant: dict, names: list[str], visible_count: int) -> None:
    """Require the real inference worker to see only the specifically approved GPU."""
    validate_selected_gpu(proposal, grant["gpu_name"], grant["gpu_count"])
    if names != [grant["gpu_name"]] or visible_count != 1:
        raise RuntimeError("Runtime GPU identity differs from the specifically approved allowlist entry")
