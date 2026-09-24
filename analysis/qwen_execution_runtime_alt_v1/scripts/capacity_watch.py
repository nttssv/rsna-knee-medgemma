#!/usr/bin/env python3
"""Read-only RunPod Secure Cloud capacity snapshot for the reviewed Qwen allowlist.

This script sends a GraphQL *query* only. It cannot start, stop, create, resize,
or otherwise change a RunPod resource, and it never writes an approval receipt.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


HERE = Path(__file__).resolve().parents[1]
PROPOSAL_PATH = HERE.parent / "qwen_execution_resource_alt_v1" / "configs" / "resource_proposal.json"
EXPECTED_PROPOSAL_SHA256 = "884dc38d2004abb44c64d2363e525969bf7f7fabd35f86eb5d03144bf34e72e2"
GRAPHQL_URL = "https://api.runpod.io/graphql"
USER_AGENT = "rsna-capacity-watch/1.0"
GPU_NAMES = (
    "NVIDIA A40",
    "NVIDIA RTX A6000",
    "NVIDIA L40",
    "NVIDIA L40S",
    "NVIDIA RTX 6000 Ada Generation",
)
STOPPED_STATES = {"EXITED", "STOPPED"}


class CapacityWatchError(RuntimeError):
    pass


def load_proposal(path: Path = PROPOSAL_PATH) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_PROPOSAL_SHA256:
        raise CapacityWatchError(
            f"resource proposal hash mismatch: expected {EXPECTED_PROPOSAL_SHA256}, got {digest}"
        )
    proposal = json.loads(raw)
    names = tuple(row["provider_name"] for row in proposal["gpu_allowlist"])
    if set(names) != set(GPU_NAMES) or len(names) != len(GPU_NAMES):
        raise CapacityWatchError("proposal GPU allowlist differs from the reviewed five-GPU set")
    return proposal, digest


def build_query(region_id: str, pod_id: str) -> tuple[str, dict[str, Any]]:
    """Return a fixed-scope read query; caller values travel only as variables."""
    query = """query QwenCapacityWatch($gpuIds: [String!]!, $region: String!, $podInput: PodFilter) {
      gpuTypes(input: {ids: $gpuIds}) {
        id displayName memoryInGb secureCloud
        lowestPrice(input: {gpuCount: 1, secureCloud: true, dataCenterId: $region}) {
          stockStatus uninterruptablePrice availableGpuCounts
        }
      }
      pod(input: $podInput) {
        id desiredStatus gpuCount containerDiskInGb volumeInGb networkVolumeId
        costPerHr adjustedCostPerHr
        machine {
          gpuTypeId gpuDisplayName gpuAvailable currentPricePerGpu costPerHr
          secureCloud dataCenterId location
        }
      }
    }"""
    return query, {
        "gpuIds": list(GPU_NAMES),
        "region": region_id,
        "podInput": {"podId": pod_id},
    }


def graphql_query(query: str, variables: dict[str, Any], api_key: str,
                   opener: Callable[..., Any] = urlopen) -> dict[str, Any]:
    if not api_key.strip():
        raise CapacityWatchError("RUNPOD_API_KEY is unset; use a read-only RunPod API key")
    body = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    request = Request(
        GRAPHQL_URL,
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                 "User-Agent": USER_AGENT},
        method="POST",
    )
    try:
        with opener(request, timeout=20) as response:
            payload = json.loads(response.read())
    except HTTPError as exc:
        raise CapacityWatchError(f"read-only RunPod query failed: HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise CapacityWatchError(f"read-only RunPod query failed: {type(exc).__name__}") from exc
    if payload.get("errors"):
        raise CapacityWatchError("RunPod GraphQL returned errors; no capacity conclusion is safe")
    if not isinstance(payload.get("data"), dict):
        raise CapacityWatchError("RunPod GraphQL response has no data object")
    return payload["data"]


def evaluate_snapshot(data: dict[str, Any], proposal: dict[str, Any], region_id: str,
                      pod_id: str, observed_at: datetime | None = None) -> dict[str, Any]:
    now = observed_at or datetime.now(timezone.utc)
    rows_by_name = {row.get("id"): row for row in data.get("gpuTypes", [])}
    candidates = []
    for allowed in proposal["gpu_allowlist"]:
        name = allowed["provider_name"]
        row = rows_by_name.get(name)
        low = (row or {}).get("lowestPrice") or {}
        counts = low.get("availableGpuCounts")
        rate = low.get("uninterruptablePrice")
        stock_status = low.get("stockStatus")
        stock = (
            stock_status in {"High", "Medium", "Low"}
            and isinstance(counts, list) and 1 in counts
        )
        stock_known = stock_status in {"High", "Medium", "Low", "None"} and isinstance(counts, list)
        rate_ok = isinstance(rate, (int, float)) and rate <= allowed["secure_compute_usd_per_hour_ceiling"]
        candidates.append({
            "gpu_name": name,
            "secure_gpu_stock_for_one": stock if stock_known else None,
            "stock_status": stock_status,
            "available_gpu_counts": counts,
            "secure_compute_usd_per_hour": rate,
            "proposal_rate_ceiling_usd_per_hour": allowed["secure_compute_usd_per_hour_ceiling"],
            "rate_within_proposal_ceiling": rate_ok,
            "region_id": region_id,
        })

    pod = data.get("pod")
    machine = (pod or {}).get("machine") or {}
    pod_values = {
        "pod_id": (pod or {}).get("id"),
        "desired_status": (pod or {}).get("desiredStatus"),
        "gpu_name": machine.get("gpuTypeId"),
        "gpu_count": (pod or {}).get("gpuCount"),
        "cloud_type": "SECURE" if machine.get("secureCloud") is True else (
            "COMMUNITY" if machine.get("secureCloud") is False else None
        ),
        "region_id": machine.get("dataCenterId"),
        "container_disk_gb": (pod or {}).get("containerDiskInGb"),
        "persistent_volume_gb": (pod or {}).get("volumeInGb"),
        "network_volume_id": (pod or {}).get("networkVolumeId"),
        "pod_cost_per_hour": (pod or {}).get("costPerHr"),
        "pod_adjusted_cost_per_hour": (pod or {}).get("adjustedCostPerHr"),
        "machine_gpu_available": machine.get("gpuAvailable"),
        "machine_current_price_per_gpu": machine.get("currentPricePerGpu"),
        "machine_cost_per_hour": machine.get("costPerHr"),
    }
    pod_exists = pod is not None and pod_values["pod_id"] == pod_id
    required_pod_keys = {"id", "desiredStatus", "gpuCount", "containerDiskInGb",
                         "volumeInGb", "networkVolumeId"}
    required_machine_keys = {"gpuTypeId", "secureCloud", "dataCenterId"}
    resource_fields_present = bool(
        pod is not None and required_pod_keys <= pod.keys()
        and machine and required_machine_keys <= machine.keys()
        and all(pod_values[k] is not None for k in (
            "gpu_name", "gpu_count", "cloud_type", "region_id", "container_disk_gb",
            "persistent_volume_gb", "desired_status",
        ))
    )
    pod_matches = bool(
        pod_exists and resource_fields_present
        and pod_values["gpu_name"] in GPU_NAMES
        and pod_values["gpu_count"] == 1
        and pod_values["cloud_type"] == "SECURE"
        and pod_values["region_id"] == region_id
        and pod_values["container_disk_gb"] == 80
        and pod_values["persistent_volume_gb"] == 0
        and pod_values["network_volume_id"] is None
        and pod_values["desired_status"] in STOPPED_STATES
    )
    matching_gpu = next((c for c in candidates if c["gpu_name"] == pod_values["gpu_name"]), None)
    pod_compute_rate = pod_values["machine_current_price_per_gpu"]
    pod_rate_ok = bool(
        matching_gpu and isinstance(pod_compute_rate, (int, float))
        and pod_compute_rate <= matching_gpu["proposal_rate_ceiling_usd_per_hour"]
    )
    if pod_matches and not pod_rate_ok:
        pod_matches = False
    aggregate_stock_for_pod_gpu = bool(
        matching_gpu and matching_gpu["secure_gpu_stock_for_one"]
        and matching_gpu["rate_within_proposal_ceiling"]
    )
    return {
        "checked_at": now.astimezone(timezone.utc).isoformat(),
        "result": "BLOCKED_CAPACITY_UNVERIFIED",
        "read_only": True,
        "resource_proposal_sha256": EXPECTED_PROPOSAL_SHA256,
        "proposal_quote_valid_until": proposal.get("quote_valid_until"),
        "proposal_quote_current": _quote_current(proposal.get("quote_valid_until"), now),
        "requested_pod_id": pod_id,
        "requested_region_id": region_id,
        "aggregate_secure_cloud_gpu_stock": candidates,
        "exact_pod": pod_values,
        "exact_pod_resource_fields_match": pod_matches,
        "aggregate_stock_matches_pod_gpu": aggregate_stock_for_pod_gpu,
        "pod_compute_rate_within_gpu_ceiling": pod_rate_ok,
        "exact_pod_resumability": "UNVERIFIED_READ_ONLY_API_HAS_NO_CAN_RESUME_RESULT",
        "can_proceed_to_approval": False,
        "explanation": (
            "Aggregate regional stock and a stopped-pod configuration snapshot do not prove "
            "that this exact stopped pod is resumable. Do not start or create a pod."
        ),
    }


def _quote_current(valid_until: str | None, now: datetime) -> bool:
    if not valid_until:
        return False
    try:
        parsed = datetime.fromisoformat(valid_until.replace("Z", "+00:00"))
    except ValueError:
        return False
    if parsed.tzinfo is None:
        return False
    return now.astimezone(timezone.utc) < parsed.astimezone(timezone.utc)


def validate_cli_value(value: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
        raise argparse.ArgumentTypeError(f"{label} must contain only letters, digits, '_' or '-'")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-id", required=True, type=lambda v: validate_cli_value(v, "region ID"),
                        help="Exact RunPod data-center ID, for example US-WA-1")
    parser.add_argument("--pod-id", required=True, type=lambda v: validate_cli_value(v, "pod ID"),
                        help="Exact existing stopped pod to inspect")
    args = parser.parse_args(argv)
    try:
        proposal, _ = load_proposal()
        query, variables = build_query(args.region_id, args.pod_id)
        data = graphql_query(query, variables, os.environ.get("RUNPOD_API_KEY", ""))
        result = evaluate_snapshot(data, proposal, args.region_id, args.pod_id)
    except CapacityWatchError as exc:
        print(json.dumps({"result": "BLOCKED_CAPACITY_UNVERIFIED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
