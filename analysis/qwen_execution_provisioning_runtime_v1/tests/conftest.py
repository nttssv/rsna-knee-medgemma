"""All provider/guard data in these tests is fabricated; no network adapters exist."""
from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/controller.py"
spec = importlib.util.spec_from_file_location("outer_controller_test", PATH)
controller = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = controller
spec.loader.exec_module(controller)


class Clock:
    def __init__(self):
        self.value = datetime(2026, 9, 24, 18, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += timedelta(seconds=seconds)


class FakeProvider:
    synthetic_only = True

    def __init__(self, clock, grant):
        self.clock, self.grant = clock, grant
        self.calls = []
        self.pod = None
        self.fail_create = False
        self.fail_resume = False
        self.stop_failures = 0

    def snapshot(self):
        value = dict(self.pod)
        value["observed_at"] = self.clock().isoformat()
        return value

    def find_by_name(self, name, *, timeout):
        self.calls.append(("find", name, timeout))
        return [self.snapshot()] if self.pod and self.pod["name"] == name else []

    def create(self, request, *, timeout):
        self.calls.append(("create", request, timeout))
        self.clock.advance(10)
        self.pod = {**request, "pod_id": "synthetic-pod", "machine_id": "synthetic-machine",
            "actual_compute_usd_per_hour": self.grant["actual_compute_usd_per_hour"],
            "actual_storage_usd_per_hour": self.grant["actual_storage_usd_per_hour"],
            "provider_state": "RUNNING", "current_total_usd_per_hour": 0.501}
        if self.fail_create:
            raise TimeoutError("Synthetic lost create response")
        return self.pod["pod_id"]

    def read(self, pod_id, *, timeout):
        self.calls.append(("read", pod_id, timeout))
        return self.snapshot()

    def stop(self, pod_id, *, timeout):
        self.calls.append(("stop", pod_id, timeout))
        self.clock.advance(1)
        if self.stop_failures:
            self.stop_failures -= 1
            raise TimeoutError("Synthetic stop timeout")
        self.pod["provider_state"] = "STOPPED"
        self.pod["current_total_usd_per_hour"] = 0.0

    def resume(self, pod_id, *, timeout):
        self.calls.append(("resume", pod_id, timeout))
        self.clock.advance(1)
        self.pod["provider_state"] = "RUNNING"
        self.pod["current_total_usd_per_hour"] = 0.501
        if self.fail_resume:
            raise TimeoutError("Synthetic ambiguous resume")


class FakeObserver:
    synthetic_only = True

    def __init__(self, provider):
        self.provider = provider
        self.calls = []

    def read(self, pod_id, *, timeout):
        self.calls.append((pod_id, timeout))
        return self.provider.snapshot()


class FakeGuard:
    synthetic_only = True

    def __init__(self):
        self.calls = []
        self.changes = {}

    def verify(self, intent, intent_sha256, *, current):
        self.calls.append((intent, intent_sha256, current))
        return {"intent_name": intent["intent_name"], "intent_sha256": intent_sha256,
            "outer_provider_deadline": intent["outer_provider_deadline"],
            "outer_watchdog_stop_at": intent["outer_watchdog_stop_at"],
            "independent_of_allocating_process": True, "alive": True,
            "command_and_environment_verified": True, "stop_capable": True,
            "pid": 999999, "verified_at": current.isoformat(), **self.changes}


class FakeInnerControl:
    synthetic_only = True

    def __init__(self):
        self.authorization_path = None

    def verify_stopped_preflight(self, grant, *, current):
        return {"pod_id": grant["pod_id"], "verified": True,
            "authorization_sha256": controller.sha(self.authorization_path)}


@pytest.fixture
def module():
    return controller


@pytest.fixture
def controller_fixture(tmp_path):
    """Factory -> (controller, provider, observer, guard, clock, outer grant)."""
    count = 0

    def make(**overrides):
        nonlocal count
        count += 1
        clock = Clock()
        grant = {**controller.FIXED, "approved": True, "approval_reference": "synthetic-only",
            "provisioning_proposal_sha256": controller.PROVISIONING_SHA,
            "reviewed_execution_plan_sha256": controller.EXECUTION_SHA,
            "prepared_plan_sha256": controller.PREPARED_SHA,
            "alternate_resource_proposal_sha256": controller.RESOURCE_SHA,
            "controller_source_sha256": controller.sha(PATH),
            "intent_name": "qwen-provision-" + "a" * 32,
            "startup_command_sha256": controller.digest(controller.STARTUP),
            "actual_compute_usd_per_hour": 0.49, "actual_storage_usd_per_hour": 0.011,
            "maximum_usd": 1.5, "maximum_provider_seconds": 3600,
            "approved_at": (clock() - timedelta(seconds=10)).isoformat(),
            "quote_observed_at": clock().isoformat(),
            "quote_valid_until": (clock() + timedelta(minutes=10)).isoformat(), **overrides}
        approval = tmp_path / f"outer-approval-{count}.json"
        controller.write_exclusive(approval, grant)
        provider = FakeProvider(clock, grant)
        observer = FakeObserver(provider)
        guard = FakeGuard()
        instance = controller.LocalController(tmp_path / f"run-{count}", approval,
            provider, observer, guard, FakeInnerControl(), clock)
        return instance, provider, observer, guard, clock, grant

    return make


@pytest.fixture
def make_inner_grant(tmp_path):
    count = 0

    def make(instance, *, grant_changes=None, observation_changes=None):
        nonlocal count
        count += 1
        now = instance.now()
        grant = {key: instance.grant[key] for key in (
            "gpu_name", "gpu_count", "cloud_type", "region", "container_disk_gb",
            "persistent_volume_gb", "network_volume_id", "actual_compute_usd_per_hour",
            "actual_storage_usd_per_hour")}
        grant.update({"approved": True, "approval_reference": "synthetic-later-stage",
            "execution_plan_sha256": controller.EXECUTION_SHA,
            "prepared_plan_sha256": controller.PREPARED_SHA,
            "resource_proposal_sha256": controller.RESOURCE_SHA,
            "pod_id": instance.pod_id, "maximum_usd": 1.0,
            "approved_at": now.isoformat(), "provider_start_requested_at": now.isoformat(),
            "provider_deadline": instance.intent["outer_provider_deadline"],
            "watchdog_stop_at": instance.intent["outer_watchdog_stop_at"]})
        grant.update(grant_changes or {})
        fields = instance.inner.alternate_gate.OBSERVED_FIELDS
        observed = {key: grant[key] for key in fields}
        observed.update({"observed_at": now.isoformat(), "provider_state": "STOPPED"})
        observed.update(observation_changes or {})
        authorization = tmp_path / f"inner-approval-{count}.json"
        observation = tmp_path / f"inner-observation-{count}.json"
        controller.write_exclusive(authorization, grant)
        controller.write_exclusive(observation, observed)
        instance.inner_control.authorization_path = authorization
        return authorization, observation

    return make
