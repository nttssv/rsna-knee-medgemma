from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qwen_guard_test", ROOT / "scripts/external_stop_guard.py")
guard = importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)
T0 = datetime(2026, 9, 25, 4, 50, tzinfo=timezone.utc)


def intent():
    return {"pod_id": "rqnenq3mpu0i2g", "pod_name": "low_brown_viper",
        "t0": T0.isoformat(), "hard_deadline": (T0 + timedelta(hours=3)).isoformat(),
        "shutdown_at": (T0 + timedelta(minutes=175)).isoformat(),
        "compute_usd_per_hour": .53, "storage_usd_per_hour": .011, "maximum_usd": 3.0,
        "execution_plan_sha256": "a" * 64, "source_sha256": "b" * 64,
        "cancellation_path": "/private/run/cancel-after-console-stop"}


def test_intent_binds_exact_pod_resource_deadline_and_budget():
    assert guard.validate_intent(intent(), now=T0 + timedelta(minutes=1)) == T0 + timedelta(minutes=175)


@pytest.mark.parametrize("key,value", [
    ("pod_id", "prior-pod"), ("pod_name", "old-name"),
    ("compute_usd_per_hour", .85), ("storage_usd_per_hour", .013),
    ("maximum_usd", 3.01), ("hard_deadline", (T0 + timedelta(hours=4)).isoformat()),
    ("shutdown_at", (T0 + timedelta(minutes=170)).isoformat()),
])
def test_guard_rejects_identity_rate_budget_or_deadline_drift(key, value):
    value_intent = intent(); value_intent[key] = value
    with pytest.raises(guard.GuardError):
        guard.validate_intent(value_intent, now=T0 + timedelta(minutes=1))


def test_stop_request_is_single_exact_bounded_mutation(tmp_path):
    key_file = tmp_path / "api.key"
    key_file.write_text("x" * 40); key_file.chmod(0o600)
    calls = []

    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, n): return b'{"data":{"podStop":{"id":"rqnenq3mpu0i2g","desiredStatus":"EXITED"}}}'

    def open_once(request, timeout):
        calls.append((request.full_url, request.method, timeout, request.data))
        return Response()

    result = guard.stop_once("rqnenq3mpu0i2g", key_file, opener=open_once)
    assert result["status"] == "stop_request_accepted"
    assert len(calls) == 1 and calls[0][0:3] == ("https://api.runpod.io/graphql", "POST", 20)
    payload = __import__("json").loads(calls[0][3])
    assert payload["query"] == 'mutation { podStop(input: { podId: "rqnenq3mpu0i2g" }) { id desiredStatus } }'


def test_wrong_id_rejected_before_provider_call(tmp_path):
    key_file = tmp_path / "api.key"
    key_file.write_text("x" * 40); key_file.chmod(0o600)
    with pytest.raises(guard.GuardError, match="Pod ID"):
        guard.stop_once("old-pod", key_file, opener=lambda *a, **k: pytest.fail("network called"))
