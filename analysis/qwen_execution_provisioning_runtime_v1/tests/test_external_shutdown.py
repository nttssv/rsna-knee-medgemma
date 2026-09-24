from datetime import timedelta
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/external_shutdown.py"
spec = importlib.util.spec_from_file_location("qwen_external_shutdown_test", SOURCE)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


@pytest.fixture
def prepared_core(controller_fixture):
    instance, provider, observer, _, clock, _ = controller_fixture()
    instance.provision_once()
    provider.calls.clear()
    observer.calls.clear()
    provider.pod.update(provider_state="RUNNING", current_total_usd_per_hour=.501)
    core = g.ShutdownCore(instance.output, instance.intent, instance.intent_sha,
        provider, observer, clock=clock, allocator_alive=lambda: False)
    return core, instance, provider, observer, clock


def test_allocator_death_stops_only_owned_exact_pod(prepared_core):
    core, _, provider, observer, _ = prepared_core
    assert core.tick() == "INDEPENDENTLY_STOPPED_ZERO_RATE"
    assert core.pod_id == "synthetic-pod"
    assert [call[0] for call in provider.calls] == ["find", "stop"]
    assert len(observer.calls) == 1
    assert core.tick() == "INDEPENDENTLY_STOPPED_ZERO_RATE"
    assert core.stop_attempts == 1


def test_shutdown_triggers_exactly_at_five_minute_reserve(prepared_core):
    core, _, provider, _, clock = prepared_core
    core.allocator_alive = lambda: True
    clock.value = core.stop_at - timedelta(microseconds=1)
    assert core.tick() == "ARMED"
    assert provider.calls == []
    clock.value = core.stop_at
    assert core.tick() == "INDEPENDENTLY_STOPPED_ZERO_RATE"
    assert provider.calls[1][2] <= 30


def test_no_attempt_no_arbitrary_stop(prepared_core):
    core, instance, provider, _, _ = prepared_core
    next(instance.output.glob("[0-9]*-create-attempt.json")).unlink()
    assert core.tick() == "NO_CREATE_ATTEMPT"
    assert provider.calls == []
    assert core.done


@pytest.mark.parametrize("matches", [0, 2, 3])
def test_ambiguous_or_absent_reconciliation_never_guesses(prepared_core, matches):
    core, _, provider, _, _ = prepared_core
    provider.find_by_name = lambda name, timeout: [provider.snapshot()] * matches
    assert core.tick() == "OWNERSHIP_UNRESOLVED"
    assert core.pod_id is None
    assert core.stop_attempts == 0
    assert not any(call[0] == "stop" for call in provider.calls)


def test_delayed_lost_create_visibility_is_read_only_reconciled(prepared_core):
    core, _, provider, _, _ = prepared_core
    original = provider.find_by_name
    provider.find_by_name = lambda name, timeout: []
    assert core.tick() == "OWNERSHIP_UNRESOLVED"
    provider.find_by_name = original
    assert core.tick() == "INDEPENDENTLY_STOPPED_ZERO_RATE"
    assert all(call[0] not in {"create", "resume"} for call in provider.calls)


@pytest.mark.parametrize("change", [
    {"gpu_name": "NVIDIA L40S"}, {"region": "US-WA-1"},
    {"actual_compute_usd_per_hour": 9.99}, {"persistent_volume_gb": 20},
])
def test_unique_owned_resource_drift_still_stops(prepared_core, change):
    core, _, provider, _, _ = prepared_core
    provider.pod.update(change)
    assert core.tick() == "INDEPENDENTLY_STOPPED_ZERO_RATE"
    assert core.stop_attempts == 1


def test_wrong_name_does_not_authorize_stop(prepared_core):
    core, _, provider, _, _ = prepared_core
    provider.find_by_name = lambda name, timeout: [{**provider.snapshot(), "name": "other-user-pod"}]
    assert core.tick() == "SHUTDOWN_UNVERIFIED"
    assert core.stop_attempts == 0


def test_bound_pod_identity_cannot_switch(prepared_core):
    core, _, provider, observer, _ = prepared_core
    provider.stop_failures = 10
    assert core.tick() == "SHUTDOWN_UNVERIFIED"
    provider.pod["pod_id"] = "different-owned-looking-pod"
    assert core.tick() == "SHUTDOWN_UNVERIFIED"
    assert core.pod_id == "synthetic-pod"
    assert core.stop_attempts == 1


def test_three_stop_attempts_maximum_and_independent_zero_billing(prepared_core):
    core, _, provider, observer, _ = prepared_core
    provider.stop_failures = 10
    for _ in range(5):
        assert core.tick() == "SHUTDOWN_UNVERIFIED"
    assert core.stop_attempts == 3
    assert len([call for call in provider.calls if call[0] == "stop"]) == 3
    assert len(observer.calls) == 5


@pytest.mark.parametrize("changes", [
    {"provider_state": "UNVERIFIED", "current_total_usd_per_hour": None},
    {"provider_state": "STOPPED", "current_total_usd_per_hour": .01},
    {"provider_state": "RUNNING", "current_total_usd_per_hour": 0},
    {"pod_id": "another-pod"}, {"name": "other-name"},
])
def test_mutation_success_is_not_independent_billing_proof(prepared_core, changes):
    core, _, provider, observer, _ = prepared_core
    observer.read = lambda pod_id, timeout: {**provider.snapshot(), **changes}
    assert core.tick() == "SHUTDOWN_UNVERIFIED"
    assert not core.done


def test_emergency_stop_survives_receipt_write_failure(prepared_core):
    core, _, provider, _, _ = prepared_core
    def broken(*args):
        raise OSError("synthetic disk full")
    core.record = broken
    assert core.tick() == "INDEPENDENTLY_STOPPED_ZERO_RATE"
    assert core.log_durable is False
    assert core.stop_attempts == 1


def test_deadline_cannot_extend_for_stop_retry(prepared_core):
    core, _, provider, _, clock = prepared_core
    clock.value = core.deadline
    assert core.tick() == "SHUTDOWN_UNVERIFIED_DEADLINE"
    assert provider.calls == []


def test_intent_tampering_invalidates_ownership(prepared_core):
    core, instance, provider, _, _ = prepared_core
    path = instance.output / "intent.json"
    changed = dict(instance.intent, maximum_usd=9)
    path.write_text(json.dumps(changed))
    assert core.tick() == "SHUTDOWN_UNVERIFIED"
    assert provider.calls == []


@pytest.mark.parametrize("field,value", [("maximum_usd", 1.51), ("maximum_usd", .01),
    ("outer_watchdog_stop_at", "2099-01-01T00:00:00+00:00")])
def test_budget_and_deadline_binding_rejected(tmp_path, field, value):
    output, intent, intent_sha, state = make_process_fixture(tmp_path)
    changed = {**intent, field: value}
    path = output / "intent.json"
    path.unlink()
    changed_sha = g.c.write_exclusive(path, changed)
    with pytest.raises(g.c.SafetyFailure):
        g.validate_intent(changed, changed_sha, path)


def make_process_fixture(tmp_path):
    output = tmp_path / "worker-output"
    output.mkdir(mode=0o700)
    now = g.utcnow()
    intent = {"intent_name": "qwen-provision-" + "b" * 32,
        "authorization_sha256": "a" * 64,
        "provisioning_proposal_sha256": g.c.PROVISIONING_SHA,
        "controller_source_sha256": g.c.sha(Path(g.c.__file__)),
        "first_create_requested_at": now.isoformat(),
        "outer_provider_deadline": (now + timedelta(seconds=3600)).isoformat(),
        "outer_watchdog_stop_at": (now + timedelta(seconds=3300)).isoformat(),
        "maximum_usd": 1.5,
        "allocation": {**g.c.FIXED, "actual_compute_usd_per_hour": .49,
                       "actual_storage_usd_per_hour": .011}}
    intent_sha = g.c.write_exclusive(output / "intent.json", intent)
    g.c.write_exclusive(output / "001-intent-name-preread.json", {
        "kind": "intent-name-preread", "recorded_at": now.isoformat(), "matches": 0})
    state = tmp_path / "synthetic-state.json"
    g.c.write_exclusive(state, {"pods": [], "stop_calls": 0})
    return output, intent, intent_sha, state


def cleanup_guard(guard):
    if guard.process is not None and guard.process.poll() is None:
        os.killpg(guard.process.pid, signal.SIGKILL)
        guard.process.wait(timeout=3)
    if guard.socket_path:
        guard.socket_path.unlink(missing_ok=True)
        try:
            guard.socket_path.parent.rmdir()
        except FileNotFoundError:
            pass


@pytest.mark.skipif(sys.platform not in {"linux", "darwin"}, reason="requires OS process attestation")
def test_real_detached_worker_requires_os_identity_and_fresh_challenge(tmp_path):
    output, intent, intent_sha, state = make_process_fixture(tmp_path)
    guard = g.ExternalGuard(output, synthetic_state=state)
    try:
        receipt = guard.verify(intent, intent_sha, current=g.utcnow())
        assert receipt["alive"] and receipt["command_and_environment_verified"]
        assert receipt["pid"] != os.getpid()
        assert os.getsid(receipt["pid"]) == receipt["pid"]
        assert receipt["synthetic_only"] is True
        guard.info["start"] = "forged-PID-start-identity"
        with pytest.raises(g.c.SafetyFailure, match="identity"):
            guard.verify(intent, intent_sha, current=g.utcnow())
    finally:
        cleanup_guard(guard)


@pytest.mark.skipif(sys.platform not in {"linux", "darwin"}, reason="requires OS process attestation")
def test_dead_worker_cannot_be_restarted_or_receipt_replayed(tmp_path):
    output, intent, intent_sha, state = make_process_fixture(tmp_path)
    guard = g.ExternalGuard(output, synthetic_state=state)
    try:
        receipt = guard.verify(intent, intent_sha, current=g.utcnow())
        os.killpg(guard.process.pid, signal.SIGKILL)
        guard.process.wait(timeout=3)
        with pytest.raises(g.c.SafetyFailure, match="died"):
            guard.verify(intent, intent_sha, current=g.utcnow())
        with pytest.raises(g.c.SafetyFailure, match="restarted"):
            guard.start(intent, intent_sha)
        assert receipt["alive"]  # Its old claim is deliberately insufficient now.
    finally:
        cleanup_guard(guard)


def test_live_worker_flag_is_false_and_no_key_read_required(tmp_path):
    output, intent, intent_sha, _ = make_process_fixture(tmp_path)
    guard = g.ExternalGuard(output, key_path=tmp_path / "does-not-exist")
    with pytest.raises(g.c.SafetyFailure, match="disabled"):
        guard.verify(intent, intent_sha, current=g.utcnow())
    assert not g.LIVE_SHUTDOWN_ENABLED
    assert guard.process is None


def test_worker_execute_cli_never_launches(tmp_path):
    result = subprocess.run([sys.executable, str(SOURCE), "--execute"], capture_output=True, text=True, timeout=5)
    assert result.returncode == 2
    assert json.loads(result.stdout)["live_shutdown_enabled"] is False


def test_preread_latency_is_inside_first_create_window(prepared_core):
    core, instance, provider, _, clock = prepared_core
    pre = next(instance.output.glob("[0-9]*-intent-name-preread.json"))
    attempt = next(instance.output.glob("[0-9]*-create-attempt.json"))
    first = g.c.time_value(instance.intent["first_create_requested_at"])
    for path, seconds in ((pre, 5), (attempt, 6)):
        data = g.c.private_json(path)
        data["recorded_at"] = (first + timedelta(seconds=seconds)).isoformat()
        path.write_text(json.dumps(data))
    assert core.tick() == "INDEPENDENTLY_STOPPED_ZERO_RATE"


def test_nonempty_preread_never_owns_a_pod(prepared_core):
    core, instance, provider, _, _ = prepared_core
    path = next(instance.output.glob("[0-9]*-intent-name-preread.json"))
    data = g.c.private_json(path)
    data["matches"] = 1
    path.write_text(json.dumps(data))
    assert core.tick() == "SHUTDOWN_UNVERIFIED"
    assert core.stop_attempts == 0


def test_private_receipt_permissions_required(prepared_core):
    core, instance, provider, _, _ = prepared_core
    path = next(instance.output.glob("[0-9]*-create-attempt.json"))
    path.chmod(0o644)
    assert core.tick() == "SHUTDOWN_UNVERIFIED"
    assert core.stop_attempts == 0


def test_cached_attempt_survives_parent_death_and_unreadable_ledger(prepared_core):
    core, instance, provider, _, _ = prepared_core
    core.allocator_alive = lambda: True
    assert core.tick() == "ARMED"
    assert core.ownership_cached is True
    for path in instance.output.glob("*.json"):
        path.unlink()
    core.allocator_alive = lambda: False
    assert core.tick() == "INDEPENDENTLY_STOPPED_ZERO_RATE"
    assert core.stop_attempts == 1


@pytest.mark.skipif(sys.platform not in {"linux", "darwin"}, reason="requires OS process attestation")
def test_challenge_acknowledges_durable_attempt_before_create_dispatch(tmp_path):
    output, intent, intent_sha, state = make_process_fixture(tmp_path)
    guard = g.ExternalGuard(output, synthetic_state=state)
    try:
        with pytest.raises(g.c.SafetyFailure, match="cached durable create ownership"):
            guard.verify(intent, intent_sha, current=g.utcnow(), require_create_attempt=True)
        g.c.write_exclusive(output / "002-create-attempt.json", {
            "kind": "create-attempt", "recorded_at": g.utcnow().isoformat(),
            "intent_sha256": intent_sha, "maximum_attempts": 1})
        receipt = guard.verify(intent, intent_sha, current=g.utcnow(), require_create_attempt=True)
        assert receipt["durable_create_attempt_verified"] is True
        (output / "002-create-attempt.json").unlink()
        receipt = guard.verify(intent, intent_sha, current=g.utcnow(), require_create_attempt=True)
        assert receipt["durable_create_attempt_verified"] is True
    finally:
        cleanup_guard(guard)
