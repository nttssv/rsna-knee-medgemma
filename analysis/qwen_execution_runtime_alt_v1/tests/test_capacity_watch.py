from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("capacity_watch_test", ROOT / "scripts/capacity_watch.py")
cw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cw)


PROPOSAL, _ = cw.load_proposal()
REGION = "US-WA-1"
POD_ID = "pod-example-123"


def fake_data(*, gpu_name="NVIDIA A40", stock="High", rate=0.49, pod_overrides=None):
    gpus = []
    for row in PROPOSAL["gpu_allowlist"]:
        current_rate = row["secure_compute_usd_per_hour_ceiling"]
        if row["provider_name"] == gpu_name:
            current_rate = rate
        gpus.append({
            "id": row["provider_name"], "displayName": row["provider_name"],
            "memoryInGb": 48, "secureCloud": True,
            "lowestPrice": {
                "stockStatus": stock, "uninterruptablePrice": current_rate,
                "availableGpuCounts": [1] if stock != "None" else [],
            },
        })
    pod = {
        "id": POD_ID, "desiredStatus": "EXITED", "gpuCount": 1,
        "containerDiskInGb": 80, "volumeInGb": 0, "networkVolumeId": None,
        "costPerHr": 0, "adjustedCostPerHr": 0,
        "machine": {
            "gpuTypeId": gpu_name, "gpuDisplayName": gpu_name,
            "gpuAvailable": 1, "currentPricePerGpu": rate, "costPerHr": 0,
            "secureCloud": True, "dataCenterId": REGION, "location": REGION,
        },
    }
    pod.update(pod_overrides or {})
    return {"gpuTypes": gpus, "pod": pod}


def assess(data=None, **kwargs):
    checked_at = kwargs.pop("observed_at", datetime(2026, 9, 25, tzinfo=timezone.utc))
    return cw.evaluate_snapshot(
        data or fake_data(), PROPOSAL, REGION, POD_ID,
        observed_at=checked_at, **kwargs,
    )


@pytest.mark.parametrize("entry", PROPOSAL["gpu_allowlist"], ids=lambda x: x["provider_name"])
def test_each_allowlisted_gpu_is_reported_when_secure_stock_and_rate_fit(entry):
    result = assess(fake_data(gpu_name=entry["provider_name"],
                              rate=entry["secure_compute_usd_per_hour_ceiling"]))
    selected = next(x for x in result["aggregate_secure_cloud_gpu_stock"]
                    if x["gpu_name"] == entry["provider_name"])
    assert selected["secure_gpu_stock_for_one"] is True
    assert selected["rate_within_proposal_ceiling"] is True
    assert result["result"] == "BLOCKED_CAPACITY_UNVERIFIED"
    assert result["can_proceed_to_approval"] is False


def test_stopped_pod_config_and_aggregate_stock_still_do_not_prove_resumability():
    result = assess()
    assert result["exact_pod_resource_fields_match"] is True
    assert result["aggregate_stock_matches_pod_gpu"] is True
    assert result["exact_pod_resumability"].startswith("UNVERIFIED")
    assert result["result"] == "BLOCKED_CAPACITY_UNVERIFIED"


@pytest.mark.parametrize("patch", [
    {"gpuCount": 2},
    {"containerDiskInGb": 100},
    {"volumeInGb": 100},
    {"networkVolumeId": "separate-volume"},
    {"desiredStatus": "RUNNING"},
])
def test_pod_configuration_drift_fails_closed(patch):
    assert assess(fake_data(pod_overrides=patch))["exact_pod_resource_fields_match"] is False


def test_machine_region_or_cloud_drift_fails_closed():
    for machine_patch in ({"dataCenterId": "US-TX-1"}, {"secureCloud": False}):
        data = fake_data()
        data["pod"]["machine"].update(machine_patch)
        assert assess(data)["exact_pod_resource_fields_match"] is False


def test_unlisted_pod_gpu_fails_exact_resource_match():
    assert assess(fake_data(pod_overrides={"machine": {"gpuTypeId": "NVIDIA H100"}}))[
        "exact_pod_resource_fields_match"
    ] is False


def test_rate_over_gpu_specific_ceiling_is_not_eligible():
    entry = PROPOSAL["gpu_allowlist"][0]
    result = assess(fake_data(gpu_name=entry["provider_name"],
                              rate=entry["secure_compute_usd_per_hour_ceiling"] + 0.01))
    row = next(x for x in result["aggregate_secure_cloud_gpu_stock"]
               if x["gpu_name"] == entry["provider_name"])
    assert row["rate_within_proposal_ceiling"] is False
    assert result["aggregate_stock_matches_pod_gpu"] is False


def test_exact_pod_machine_rate_is_also_checked():
    data = fake_data()
    data["pod"]["machine"]["currentPricePerGpu"] = 0.50
    result = assess(data)
    assert result["pod_compute_rate_within_gpu_ceiling"] is False
    assert result["exact_pod_resource_fields_match"] is False


def test_quote_freshness_is_reported_without_refreshing_or_mutating_proposal():
    result = assess(observed_at=datetime(2026, 9, 26, tzinfo=timezone.utc))
    assert result["proposal_quote_current"] is False
    assert result["proposal_quote_valid_until"] == PROPOSAL["quote_valid_until"]


def test_no_stock_is_not_reported_as_available():
    result = assess(fake_data(stock="None"))
    row = next(x for x in result["aggregate_secure_cloud_gpu_stock"]
               if x["gpu_name"] == "NVIDIA A40")
    assert row["secure_gpu_stock_for_one"] is False


def test_missing_regional_stock_is_unknown_not_a_reported_shortage():
    data = fake_data()
    data["gpuTypes"][0]["lowestPrice"] = None
    result = assess(data)
    assert result["aggregate_secure_cloud_gpu_stock"][0]["secure_gpu_stock_for_one"] is None
    assert result["aggregate_stock_matches_pod_gpu"] is False
    assert result["can_proceed_to_approval"] is False


def test_query_is_read_only_and_binds_only_allowlisted_gpu_and_exact_pod_variables():
    query, variables = cw.build_query(REGION, POD_ID)
    assert query.lstrip().startswith("query ")
    assert "mutation" not in query.lower()
    assert "podResume" not in query and "podFindAndDeploy" not in query
    assert "gpuTypes" in query and "pod(input: $podInput)" in query
    assert variables == {"gpuIds": list(cw.GPU_NAMES), "region": REGION,
                         "podInput": {"podId": POD_ID}}


def test_proposal_hash_is_pinned_and_loads_five_allowlisted_gpus():
    proposal, digest = cw.load_proposal()
    assert digest == cw.EXPECTED_PROPOSAL_SHA256
    assert len(proposal["gpu_allowlist"]) == 5


def test_graphql_requires_key_without_disclosing_it():
    query, variables = cw.build_query(REGION, POD_ID)
    with pytest.raises(cw.CapacityWatchError, match="RUNPOD_API_KEY"):
        cw.graphql_query(query, variables, "")


def test_graphql_errors_fail_closed():
    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def read(self): return json.dumps({"errors": [{"message": "denied"}]}).encode()

    query, variables = cw.build_query(REGION, POD_ID)
    with pytest.raises(cw.CapacityWatchError, match="returned errors"):
        cw.graphql_query(query, variables, "dummy-read-only-key", opener=lambda *_a, **_k: Response())


def test_http_request_uses_application_identity_and_header_auth_only():
    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def read(self): return b'{"data": {"pod": null, "gpuTypes": []}}'

    def opener(request, timeout):
        assert request.full_url == cw.GRAPHQL_URL
        assert request.get_header("User-agent") == "rsna-capacity-watch/1.0"
        assert request.get_header("Authorization") == "Bearer dummy-read-only-key"
        assert b"dummy-read-only-key" not in request.data
        assert json.loads(request.data)["query"].lstrip().startswith("query ")
        assert request.method == "POST" and timeout == 20
        return Response()

    query, variables = cw.build_query(REGION, POD_ID)
    assert cw.graphql_query(query, variables, "dummy-read-only-key", opener=opener)["pod"] is None


def test_http_errors_report_status_without_secret_or_response_body():
    from urllib.error import HTTPError
    import io

    def opener(request, timeout):
        raise HTTPError(request.full_url, 403, "dummy-read-only-key", {},
                        io.BytesIO(b"dummy-read-only-key"))

    query, variables = cw.build_query(REGION, POD_ID)
    with pytest.raises(cw.CapacityWatchError) as caught:
        cw.graphql_query(query, variables, "dummy-read-only-key", opener=opener)
    assert str(caught.value) == "read-only RunPod query failed: HTTP 403"
