"""Integration boundary attacks using private synthetic receipts and local processes.

These tests never construct a RunPod key, send HTTP, start a pod, or load a model.
"""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(f"live_integration_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(SCRIPTS))
    return mod


@pytest.fixture
def policy():
    return load("live_policy")


def test_once_claim_survives_new_controller_object_without_result(tmp_path, policy):
    """A lost process after durable attempt never grants a second dispatch."""
    name = "qwen-provision-" + "b" * 32
    bindings = {"authorization_sha256": "a" * 64, "source_sha256": "b" * 64}
    first = policy.OnceLedger(tmp_path / "ledger", name, bindings)
    first.claim("create")
    reopened = policy.OnceLedger(tmp_path / "ledger", name, bindings)
    with pytest.raises(Exception):
        reopened.claim("create")
    # The only other allowed mutation is one later resume, with its own claim.
    reopened.claim("resume")
    again = policy.OnceLedger(tmp_path / "ledger", name, bindings)
    with pytest.raises(Exception):
        again.claim("resume")


def test_once_claim_remains_consumed_after_claiming_process_dies(tmp_path, policy):
    name = "qwen-provision-" + "9" * 32
    bindings = {"authorization_sha256": "9" * 64}
    child = """import json, os, sys
sys.path.insert(0, sys.argv[1])
from pathlib import Path
from live_policy import OnceLedger
OnceLedger(Path(sys.argv[2]), sys.argv[3], json.loads(sys.argv[4])).claim('create')
os._exit(19)
"""
    result = subprocess.run([sys.executable, "-c", child, str(SCRIPTS),
        str(tmp_path / "ledger"), name, json.dumps(bindings)],
        capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode == 19, result.stderr
    ledger = policy.OnceLedger(tmp_path / "ledger", name, bindings)
    with pytest.raises(Exception):
        ledger.claim("create")


def test_once_claim_contended_across_independent_objects_has_one_winner(tmp_path, policy):
    name = "qwen-provision-" + "c" * 32
    bindings = {"authorization_sha256": "c" * 64}
    policy.OnceLedger(tmp_path / "ledger", name, bindings)

    def claim():
        try:
            policy.OnceLedger(tmp_path / "ledger", name, bindings).claim("create")
        except Exception:
            return False
        return True

    with ThreadPoolExecutor(max_workers=6) as pool:
        assert sum(pool.map(lambda _: claim(), range(6))) == 1


@pytest.mark.parametrize("operation", ["create-again", "stop", "delete", "restart", "fallback", "", "../create"])
def test_once_ledger_does_not_offer_replacement_or_unreviewed_operation(tmp_path, policy, operation):
    ledger = policy.OnceLedger(tmp_path / "ledger", "qwen-provision-" + "d" * 32,
                               {"authorization_sha256": "d" * 64})
    with pytest.raises(Exception):
        ledger.claim(operation)


def test_changed_grant_cannot_reuse_intent_ledger(tmp_path, policy):
    name = "qwen-provision-" + "e" * 32
    policy.OnceLedger(tmp_path / "ledger", name, {"authorization_sha256": "e" * 64}).claim("create")
    with pytest.raises(Exception):
        policy.OnceLedger(tmp_path / "ledger", name, {"authorization_sha256": "f" * 64})


@pytest.mark.parametrize("operation", ["provisioning_enabled", "resume_enabled", "execution_enabled"])
def test_committed_live_switches_block_before_any_provider_construction(policy, operation):
    cfg = json.loads((ROOT / "configs/runtime.json").read_text())
    assert cfg[operation] is False
    with pytest.raises(Exception):
        policy.require_live(operation)


def test_factory_disabled_before_inspecting_credential_or_adapter(module, monkeypatch):
    calls = []

    def fail_private_read(*args, **kwargs):
        calls.append("private-read")
        raise AssertionError("Disabled factory attempted private credential/approval access")

    monkeypatch.setattr(module, "private_json", fail_private_read)
    # With no arguments, a correct gate still runs before it inspects provider inputs.
    with pytest.raises(Exception) as caught:
        module.create_live_controller()
    assert not isinstance(caught.value, TypeError), "Factory resolved required live arguments before its disabled gate"
    assert calls == []


def test_source_bundle_validates_every_runtime_script(policy):
    digest = policy.verify_source_bundle()
    assert isinstance(digest, str) and len(digest) == 64
    with pytest.raises(Exception):
        policy.verify_source_bundle("0" * 64)


def test_unknown_actual_billing_cannot_be_relabelled_zero(controller_fixture, module):
    ctl, provider, observer, *_ = controller_fixture()
    original = observer.read

    def unreadable_billing(pod_id, *, timeout):
        value = original(pod_id, timeout=timeout)
        value["current_total_usd_per_hour"] = None
        return value

    observer.read = unreadable_billing
    with pytest.raises(module.SafetyFailure, match="Shutdown unverified"):
        ctl.provision_once()
    assert sum(call[0] == "create" for call in provider.calls) == 1
    assert sum(call[0] == "stop" for call in provider.calls) == 3
    assert ctl.phase == "FAILED_SHUTDOWN_UNVERIFIED"
    with pytest.raises(module.SafetyFailure):
        ctl.provision_once()


def test_exact_resource_failure_keeps_cost_unknown_after_emergency_stop(controller_fixture, module):
    ctl, provider, *_ = controller_fixture()
    original = provider.read

    def changed_rate(pod_id, *, timeout):
        value = original(pod_id, timeout=timeout)
        value["actual_compute_usd_per_hour"] = 4.90
        return value

    provider.read = changed_rate
    with pytest.raises(module.SafetyFailure, match="rate"):
        ctl.provision_once()
    assert provider.pod["provider_state"] == "STOPPED"
    assert ctl.cost_bound_verified is False
    stopped = [json.loads(p.read_text()) for p in ctl.output.glob("*-independently-stopped.json")]
    assert stopped and stopped[-1]["cost_bound_status"] == "UNVERIFIED"
    assert stopped[-1]["accrued_upper_usd"] is None


def test_inner_handoff_preserves_frozen_disabled_gate_before_private_inputs(tmp_path, monkeypatch):
    handoff = load("handoff")
    runner = handoff.InnerRunnerHandoff(**{name: tmp_path / f"absent-{name}" for name in (
        "prepared", "cache", "output", "authorization", "watchdog_receipt",
        "stop_preflight", "provider_observation")})
    private_reads = []

    def forbidden_private_read(*args, **kwargs):
        private_reads.append(True)
        raise AssertionError("Frozen disabled gate must precede private inputs")

    monkeypatch.setattr(handoff.outer, "private_json", forbidden_private_read)
    with pytest.raises(PermissionError, match="Frozen inner execution remains disabled"):
        runner.frozen_gate_ready()
    assert private_reads == []
    assert not any(tmp_path.iterdir())


@pytest.mark.skipif(sys.platform not in {"linux", "darwin"}, reason="Requires POSIX process identity")
def test_detached_shutdown_survives_allocator_process_group_death(tmp_path):
    """The independent worker stops a local synthetic pod after its allocator dies."""
    import os
    import signal
    import time

    out = tmp_path / "synthetic-session"
    state = tmp_path / "synthetic-provider.json"
    helper = r'''import json, sys, time
from pathlib import Path
from datetime import timedelta
sys.path.insert(0, sys.argv[1])
import controller as c
from external_shutdown import ExternalGuard, utcnow
out, state = Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(mode=0o700)
first=utcnow(); deadline=first+timedelta(seconds=600)
name='qwen-provision-'+'8'*32
intent={'intent_name':name, 'first_create_requested_at':first.isoformat(),
    'outer_provider_deadline':deadline.isoformat(),
    'outer_watchdog_stop_at':(deadline-timedelta(seconds=300)).isoformat(),
    'maximum_usd':1.5, 'provisioning_proposal_sha256':c.PROVISIONING_SHA,
    'controller_source_sha256':c.sha(Path(c.__file__)),
    'allocation':dict(c.FIXED,actual_compute_usd_per_hour=.49,actual_storage_usd_per_hour=.011)}
intent_sha=c.write_exclusive(out/'intent.json',intent)
c.write_exclusive(out/'001-intent-name-preread.json',
    {'kind':'intent-name-preread','matches':0,'recorded_at':first.isoformat()})
c.write_exclusive(out/'002-create-attempt.json',
    {'kind':'create-attempt','intent_sha256':intent_sha,'maximum_attempts':1,'recorded_at':utcnow().isoformat()})
c.write_exclusive(state, {'pods':[{'pod_id':'synthetic-owned-pod','name':name,
    'provider_state':'RUNNING','current_total_usd_per_hour':.501}], 'stop_calls':0})
guard=ExternalGuard(out,synthetic_state=state)
receipt=guard.verify(intent,intent_sha,current=utcnow())
print(json.dumps({'pid':receipt['pid']}),flush=True)
while True: time.sleep(1)
'''
    allocator = subprocess.Popen([sys.executable, "-c", helper, str(SCRIPTS), str(out), str(state)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    worker_pid = None
    try:
        import select
        ready, _, _ = select.select([allocator.stdout], [], [], 8)
        assert ready, "Synthetic detached guard never became ready"
        line = allocator.stdout.readline()
        if not line:
            _, stderr = allocator.communicate(timeout=3)
            pytest.fail("Synthetic guard startup failed: " + stderr)
        worker_pid = json.loads(line)["pid"]
        assert os.getpgid(worker_pid) == worker_pid
        assert os.getpgid(allocator.pid) == allocator.pid
        assert worker_pid != allocator.pid
        os.killpg(allocator.pid, signal.SIGKILL)
        allocator.wait(timeout=3)
        end = time.monotonic() + 6
        while time.monotonic() < end:
            value = json.loads(state.read_text())
            if value["stop_calls"]:
                break
            time.sleep(.05)
        assert value["stop_calls"] == 1
        assert value["pods"][0]["provider_state"] == "STOPPED"
        assert value["pods"][0]["current_total_usd_per_hour"] == 0
        # Neither the allocator nor guard has any synthetic create/resume method.
        assert value["pods"][0]["pod_id"] == "synthetic-owned-pod"
    finally:
        if allocator.poll() is None:
            os.killpg(allocator.pid, signal.SIGKILL)
            allocator.wait(timeout=3)
        for stream in (allocator.stdout, allocator.stderr):
            if stream:
                stream.close()
        if worker_pid:
            config_path = out / "external-guard/worker.json"
            socket_path = Path(json.loads(config_path.read_text())["socket_path"])
            cleanup_until = time.monotonic() + 2
            while socket_path.exists() and time.monotonic() < cleanup_until:
                time.sleep(.02)
            try:
                os.killpg(worker_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            socket_path.unlink(missing_ok=True)
            if socket_path.parent.exists():
                socket_path.parent.rmdir()


def test_external_guard_accepts_controller_preread_latency_without_resetting_clock(controller_fixture):
    """The initial provider read runs inside the immutable outer time window."""
    guard_module = load("external_shutdown")
    ctl, provider, observer, _, clock, _ = controller_fixture()
    original = provider.find_by_name

    def delayed_preread(name, *, timeout):
        clock.advance(10)
        return original(name, timeout=timeout)

    provider.find_by_name = delayed_preread
    ctl.provision_once()
    core = guard_module.ShutdownCore(ctl.output, ctl.intent, ctl.intent_sha, provider,
        observer, clock=clock, allocator_alive=lambda: False)
    assert core.ownership_evidence() is True
    # Cleanup is idempotent over the same pod; no fresh creation or resumed budget.
    first = ctl.intent["first_create_requested_at"]
    assert core.tick() == "INDEPENDENTLY_STOPPED_ZERO_RATE"
    assert ctl.intent["first_create_requested_at"] == first
    assert sum(call[0] == "create" for call in provider.calls) == 1
