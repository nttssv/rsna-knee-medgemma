"""Adversarial lifecycle tests; every provider/control object is synthetic."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("outer_adversarial_controller", ROOT / "scripts/controller.py")
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


def private(path, value):
    path.write_text(json.dumps(value))
    path.chmod(0o600)
    return path


class Clock:
    def __init__(self):
        self.value = datetime(2026, 9, 24, 18, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += timedelta(seconds=seconds)


class Provider:
    synthetic_only = True

    def __init__(self, clock, grant):
        self.clock, self.grant = clock, grant
        self.calls = []
        self.state = "RUNNING"
        self.machine_id = "synthetic-machine-original"
        self.create_error = None
        self.reconciled = []
        self.read_drift = {}
        self.replace_machine_on_resume = False

    def row(self):
        return {
            **deepcopy(controller.FIXED), "name": self.grant["intent_name"],
            "pod_id": "synthetic-created-pod", "machine_id": self.machine_id,
            "actual_compute_usd_per_hour": self.grant["actual_compute_usd_per_hour"],
            "actual_storage_usd_per_hour": self.grant["actual_storage_usd_per_hour"],
            "provider_state": self.state, "observed_at": self.clock().isoformat(),
            "current_total_usd_per_hour": 0 if self.state == "STOPPED" else 0.501,
        }

    def find_by_name(self, name, *, timeout):
        self.calls.append(("find", name))
        return deepcopy(self.reconciled) if any(c[0] == "create" for c in self.calls) else []

    def create(self, request, *, timeout):
        self.calls.append(("create", deepcopy(request)))
        if self.create_error:
            raise self.create_error
        return "synthetic-created-pod"

    def read(self, pod_id, *, timeout):
        self.calls.append(("read", pod_id))
        return {**self.row(), **deepcopy(self.read_drift)}

    def stop(self, pod_id, *, timeout):
        self.calls.append(("stop", pod_id))
        self.state = "STOPPED"

    def resume(self, pod_id, *, timeout):
        self.calls.append(("resume", pod_id))
        self.state = "RUNNING"
        if self.replace_machine_on_resume:
            self.machine_id = "synthetic-replacement-machine"


class Observer:
    synthetic_only = True

    def __init__(self, provider):
        self.provider = provider
        self.drift = {}

    def read(self, pod_id, *, timeout):
        return {**self.provider.row(), **deepcopy(self.drift)}


class Guard:
    synthetic_only = True

    def __init__(self, clock):
        self.clock = clock
        self.drift = {}
        self.advance_seconds = 0

    def verify(self, intent, intent_sha256, *, current):
        self.clock.advance(self.advance_seconds)
        return {
            "intent_sha256": intent_sha256, "intent_name": intent["intent_name"],
            "outer_provider_deadline": intent["outer_provider_deadline"],
            "outer_watchdog_stop_at": intent["outer_watchdog_stop_at"],
            "independent_of_allocating_process": True, "alive": True,
            "command_and_environment_verified": True, "stop_capable": True,
            "pid": 2147483000, "verified_at": self.clock().isoformat(), **self.drift,
        }


class InnerControl:
    synthetic_only = True

    def __init__(self, clock):
        self.clock, self.grant_path = clock, None
        self.advance_seconds = 0

    def verify_stopped_preflight(self, grant, *, current):
        self.clock.advance(self.advance_seconds)
        return {"pod_id": grant["pod_id"], "verified": True,
                "authorization_sha256": controller.sha(self.grant_path)}


@pytest.fixture
def world(tmp_path):
    clock = Clock()
    grant = {
        "approved": True, "approval_reference": "SYNTHETIC ADVERSARIAL TEST ONLY",
        "provisioning_proposal_sha256": controller.PROVISIONING_SHA,
        "reviewed_execution_plan_sha256": controller.EXECUTION_SHA,
        "prepared_plan_sha256": controller.PREPARED_SHA,
        "alternate_resource_proposal_sha256": controller.RESOURCE_SHA,
        "controller_source_sha256": controller.sha(Path(controller.__file__)),
        "intent_name": "qwen-provision-" + "f" * 32,
        **deepcopy(controller.FIXED), "startup_command_sha256": controller.digest(controller.STARTUP),
        "actual_compute_usd_per_hour": 0.49, "actual_storage_usd_per_hour": 0.011,
        "maximum_usd": 1.5, "maximum_provider_seconds": 3600,
        "approved_at": (clock() - timedelta(seconds=2)).isoformat(),
        "quote_observed_at": (clock() - timedelta(seconds=1)).isoformat(),
        "quote_valid_until": (clock() + timedelta(hours=1)).isoformat(),
    }
    path = private(tmp_path / "synthetic-outer-approval.json", grant)
    provider, guard, inner_control = Provider(clock, grant), Guard(clock), InnerControl(clock)
    observer = Observer(provider)
    ctl = controller.LocalController(tmp_path / "synthetic-receipts", path, provider,
                                     observer, guard, inner_control, clock)
    return ctl, provider, observer, guard, inner_control, clock, grant, tmp_path


def resume_inputs(world, *, maximum=1.4):
    ctl, _, _, _, control, clock, _, tmp_path = world
    grant = {
        "approved": True, "approval_reference": "SYNTHETIC INNER TEST ONLY",
        "execution_plan_sha256": controller.EXECUTION_SHA,
        "prepared_plan_sha256": controller.PREPARED_SHA,
        "resource_proposal_sha256": controller.RESOURCE_SHA,
        "pod_id": ctl.pod_id,
        **{key: ctl.grant[key] for key in ctl.inner.alternate_gate.OBSERVED_FIELDS if key != "pod_id"},
        "maximum_usd": maximum, "approved_at": clock().isoformat(),
        "provider_start_requested_at": clock().isoformat(),
        "provider_deadline": ctl.intent["outer_provider_deadline"],
        "watchdog_stop_at": ctl.intent["outer_watchdog_stop_at"],
    }
    observed = {key: grant[key] for key in ctl.inner.alternate_gate.OBSERVED_FIELDS}
    observed.update(observed_at=clock().isoformat(), provider_state="STOPPED")
    grant_path = private(tmp_path / "synthetic-inner-approval.json", grant)
    control.grant_path = grant_path
    observation_path = private(tmp_path / "synthetic-observation.json", observed)
    return grant_path, observation_path, grant, observed


@pytest.mark.parametrize("matching_count", [0, 2])
def test_ambiguous_create_never_dispatches_a_second_request(world, matching_count):
    ctl, provider, *_ = world
    provider.create_error = TimeoutError("synthetic lost response")
    provider.reconciled = [provider.row() for _ in range(matching_count)]
    with pytest.raises(controller.SafetyFailure, match="identity unresolved"):
        ctl.provision_once()
    with pytest.raises(controller.SafetyFailure, match="No second"):
        ctl.provision_once()
    assert sum(kind == "create" for kind, _ in provider.calls) == 1
    assert not any(kind in {"resume", "stop"} for kind, _ in provider.calls)


@pytest.mark.parametrize("lost_response", [False, True])
def test_resource_failure_stops_the_known_created_pod(world, lost_response):
    ctl, provider, *_ = world
    if lost_response:
        provider.create_error = TimeoutError("synthetic lost response")
        provider.reconciled = [{**provider.row(), "container_disk_gb": 81}]
    else:
        provider.read_drift = {"container_disk_gb": 81}
    with pytest.raises(controller.SafetyFailure, match="container_disk_gb"):
        ctl.provision_once()
    assert ("stop", "synthetic-created-pod") in provider.calls
    assert provider.state == "STOPPED"
    assert sum(kind == "create" for kind, _ in provider.calls) == 1


def test_changed_physical_machine_after_resume_is_stopped_without_fallback(world):
    ctl, provider, *_ = world
    ctl.provision_once()
    args = resume_inputs(world)
    provider.replace_machine_on_resume = True
    with pytest.raises(controller.SafetyFailure, match="physical host"):
        ctl.resume_once(*args[:2])
    assert provider.state == "STOPPED"
    assert sum(kind == "resume" for kind, _ in provider.calls) == 1
    assert all(target == "synthetic-created-pod" for kind, target in provider.calls if kind == "stop")
    with pytest.raises(controller.SafetyFailure):
        ctl.resume_once(*args[:2])
    assert sum(kind == "resume" for kind, _ in provider.calls) == 1


@pytest.mark.parametrize("field,value", [
    ("alive", False), ("command_and_environment_verified", False),
    ("independent_of_allocating_process", False), ("stop_capable", False),
    ("pid", None), ("pid", 0), ("pid", True), ("intent_sha256", "0" * 64),
    ("verified_at", "2026-09-24T17:59:54+00:00"),
])
def test_unready_or_unidentified_watchdog_prevents_create(world, field, value):
    ctl, provider, _, guard, *_ = world
    guard.drift[field] = value
    with pytest.raises(controller.SafetyFailure):
        ctl.provision_once()
    assert not any(kind == "create" for kind, _ in provider.calls)


@pytest.mark.parametrize("stage", ["preflight", "guard"])
def test_preflight_delay_expires_resume_dispatch_timestamp(world, stage):
    ctl, provider, _, guard, control, *_ = world
    ctl.provision_once()
    args = resume_inputs(world)
    (control if stage == "preflight" else guard).advance_seconds = 6
    with pytest.raises(controller.SafetyFailure):
        ctl.resume_once(*args[:2])
    assert not any(kind == "resume" for kind, _ in provider.calls)


@pytest.mark.parametrize("field,value", [
    ("observed_at", "2026-09-24T17:49:59+00:00"),
    ("observed_at", "2026-09-24T18:00:01+00:00"),
    ("observed_at", "2026-09-24T18:00:00"),
    ("provider_state", "RUNNING"), ("pod_id", "synthetic-other-pod"),
    ("region", "US-WA-1"), ("network_volume_id", "synthetic-volume"),
])
def test_stale_or_forged_stopped_observation_never_resumes(world, field, value):
    ctl, provider, *_ = world
    ctl.provision_once()
    grant_path, observation_path, _, observed = resume_inputs(world)
    private(observation_path, {**observed, field: value})
    with pytest.raises((controller.SafetyFailure, PermissionError, ValueError)):
        ctl.resume_once(grant_path, observation_path)
    assert not any(kind == "resume" for kind, _ in provider.calls)


def test_subcent_prior_cost_is_not_rounded_down_to_admit_overbudget_resume(world):
    ctl, provider, *_ = world
    ctl.provision_once()
    ctl.accrued_upper = Decimal("0.000000000000000001")
    args = resume_inputs(world, maximum=1.5)
    with pytest.raises(controller.SafetyFailure, match="accrued"):
        ctl.resume_once(*args[:2])
    assert not any(kind == "resume" for kind, _ in provider.calls)


@pytest.mark.parametrize("field,value", [
    ("maximum_provider_seconds", 3601), ("maximum_provider_seconds", 300),
    ("maximum_usd", 1.5000000001), ("maximum_usd", 0.5019999999999999),
    ("actual_compute_usd_per_hour", 0.4900000000000001),
    ("actual_storage_usd_per_hour", 0.012000000000001),
])
def test_outer_window_and_exact_cost_bound_cannot_be_rounded_or_extended(world, field, value):
    _, _, _, _, _, clock, grant, _ = world
    with pytest.raises(controller.SafetyFailure):
        controller.validate_outer_approval({**grant, field: value}, clock())


def test_resume_deadline_cannot_reset_creation_clock(world):
    ctl, provider, *_ = world
    ctl.provision_once()
    grant_path, observation_path, grant, _ = resume_inputs(world)
    grant["provider_deadline"] = (controller.time_value(grant["provider_deadline"]) + timedelta(seconds=1)).isoformat()
    grant["watchdog_stop_at"] = (controller.time_value(grant["watchdog_stop_at"]) + timedelta(seconds=1)).isoformat()
    private(grant_path, grant)
    with pytest.raises((controller.SafetyFailure, PermissionError)):
        ctl.resume_once(grant_path, observation_path)
    assert not any(kind == "resume" for kind, _ in provider.calls)


def test_tampered_durable_intent_blocks_resume(world):
    ctl, provider, *_ = world
    ctl.provision_once()
    args = resume_inputs(world)
    intent_path = ctl.output / "intent.json"
    value = json.loads(intent_path.read_text())
    value["maximum_usd"] = 99
    private(intent_path, value)
    with pytest.raises(controller.SafetyFailure):
        ctl.resume_once(*args[:2])
    assert not any(kind == "resume" for kind, _ in provider.calls)


def test_receipt_write_failure_cannot_prevent_stopping_known_created_pod(world, monkeypatch):
    ctl, provider, *_ = world
    original = ctl.event

    def fail_stop_receipt(kind, fields=None):
        if kind.startswith("stop-"):
            raise OSError("synthetic disk full")
        return original(kind, fields)

    monkeypatch.setattr(ctl, "event", fail_stop_receipt)
    try:
        ctl.provision_once()
    except (OSError, controller.SafetyFailure):
        pass
    assert ("stop", "synthetic-created-pod") in provider.calls
    assert provider.state == "STOPPED"


def test_stop_verification_rejects_stale_snapshot_after_all_three_attempts(world):
    ctl, provider, observer, *_ = world
    observer.drift["observed_at"] = "2026-09-24T17:59:54+00:00"
    with pytest.raises(controller.SafetyFailure, match="Shutdown unverified"):
        ctl.provision_once()
    assert sum(kind == "stop" for kind, _ in provider.calls) == 3
    assert ctl.phase == "FAILED_SHUTDOWN_UNVERIFIED"
