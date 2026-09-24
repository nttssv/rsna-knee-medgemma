from datetime import timedelta
from decimal import Decimal
import os
import subprocess
import sys

import pytest


def test_create_stop_is_one_request_and_never_inference(controller_fixture, module):
    instance, provider, observer, guard, clock, grant = controller_fixture()
    pod = instance.provision_once()
    assert pod == "synthetic-pod"
    assert instance.phase == "STOPPED_PREPARATION_COMPLETE"
    assert [item[0] for item in provider.calls] == ["find", "create", "read", "stop"]
    assert len(observer.calls) == 1
    assert len(guard.calls) == 1
    assert provider.pod["provider_state"] == "STOPPED"
    assert provider.pod["image_ref"] == module.IMAGE
    assert provider.pod["startup_command"] == ["/bin/sleep", "infinity"]
    assert provider.pod["network_volume_id"] is None
    assert instance.accrued_upper == module.accrued_ceiling(
        module.time_value(instance.intent["first_create_requested_at"]), clock())
    events = sorted(instance.output.glob("*.json"))
    assert any("create-attempt" in path.name for path in events)
    assert any("allocation-bound" in path.name for path in events)
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in events)
    assert instance.output.stat().st_mode & 0o777 == 0o700
    assert not hasattr(instance, "run_inference")


def test_one_later_resume_keeps_outer_clock_and_stops(controller_fixture, make_inner_grant, module):
    instance, provider, observer, guard, clock, _ = controller_fixture()
    instance.provision_once()
    before = dict(instance.intent)
    clock.advance(10)
    approval, observation = make_inner_grant(instance)
    instance.resume_once(approval, observation)
    assert instance.phase == "STOPPED_LOCAL_LIFECYCLE_COMPLETE"
    assert instance.intent == before
    assert len([call for call in provider.calls if call[0] == "create"]) == 1
    assert len([call for call in provider.calls if call[0] == "resume"]) == 1
    assert len([call for call in provider.calls if call[0] == "stop"]) == 2
    assert len(guard.calls) == 2
    with pytest.raises(module.SafetyFailure):
        instance.resume_once(approval, observation)
    with pytest.raises(module.SafetyFailure):
        instance.provision_once()


def test_fresh_read_after_small_observer_latency_passes(controller_fixture, make_inner_grant):
    instance, provider, observer, _, clock, _ = controller_fixture()
    instance.provision_once()
    approval, observation = make_inner_grant(instance)
    read = observer.read

    def delayed_read(pod_id, *, timeout):
        clock.advance(1)
        return read(pod_id, timeout=timeout)

    observer.read = delayed_read
    instance.resume_once(approval, observation)
    assert instance.phase == "STOPPED_LOCAL_LIFECYCLE_COMPLETE"


@pytest.mark.parametrize("field,value", [
    ("cloud_type", "COMMUNITY"), ("region", "US-WA-1"),
    ("gpu_name", "NVIDIA L40"), ("gpu_count", 2), ("gpu_count", True),
    ("gpu_vram_gb", 40), ("container_disk_gb", 100),
    ("persistent_volume_gb", 1), ("network_volume_id", "unapproved"),
    ("actual_compute_usd_per_hour", .491), ("actual_compute_usd_per_hour", 0),
    ("actual_storage_usd_per_hour", .013), ("maximum_usd", 1.51),
    ("maximum_usd", .1), ("maximum_provider_seconds", 3601),
    ("maximum_provider_seconds", 300), ("maximum_provider_seconds", True),
    ("image_ref", "runpod/pytorch:latest"), ("template_id", "mutable-template"),
    ("startup_command", ["python", "train.py"]),
    ("startup_command_sha256", "0" * 64), ("controller_source_sha256", "0" * 64),
    ("approved", False), ("intent_name", "low-entropy-name"),
    ("prepared_plan_sha256", "0" * 64),
    ("reviewed_execution_plan_sha256", "0" * 64),
    ("alternate_resource_proposal_sha256", "0" * 64),
    ("provisioning_proposal_sha256", "0" * 64),
])
def test_outer_approval_drift_fails_before_adapters(controller_fixture, module, field, value):
    with pytest.raises(module.SafetyFailure):
        controller_fixture(**{field: value})


def test_expired_quote_never_creates(controller_fixture, module):
    instance, provider, _, _, clock, _ = controller_fixture()
    clock.advance(601)
    with pytest.raises(module.SafetyFailure, match="quote"):
        instance.provision_once()
    assert not any(call[0] == "create" for call in provider.calls)


def test_private_approval_permissions_symlinks_and_duplicate_keys(tmp_path, module):
    path = tmp_path / "input.json"
    path.write_text('{"a": 1, "a": 2}')
    path.chmod(0o600)
    with pytest.raises(module.SafetyFailure, match="Duplicate"):
        module.private_json(path)
    path.write_text('{"a": 1}')
    path.chmod(0o644)
    with pytest.raises(module.SafetyFailure, match="0600"):
        module.private_json(path)
    path.chmod(0o600)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(OSError):
        module.private_json(link)


def test_exclusive_durable_receipts_cannot_overwrite(tmp_path, module):
    path = tmp_path / "receipt.json"
    first = module.write_exclusive(path, {"one": True})
    with pytest.raises(FileExistsError):
        module.write_exclusive(path, {"two": True})
    assert module.sha(path) == first


@pytest.mark.parametrize("amount", [float("nan"), float("inf"), -1, True, "0.49", None])
def test_numeric_amounts_fail_closed(module, amount):
    with pytest.raises(module.SafetyFailure):
        module.money(amount)


def test_stop_retries_only_shutdown(controller_fixture):
    instance, provider, *_ = controller_fixture()
    provider.stop_failures = 2
    instance.provision_once()
    assert len([call for call in provider.calls if call[0] == "stop"]) == 3
    assert len([call for call in provider.calls if call[0] == "create"]) == 1


def test_cli_refuses_live_even_with_execute(module):
    for flags in ([], ["--execute"]):
        result = subprocess.run([sys.executable, str(module.Path(module.__file__)), *flags],
                                capture_output=True, text=True, check=False)
        assert result.returncode == 2
        assert '"execution_enabled": false' in result.stdout
        assert '"provider_calls": 0' in result.stdout


def test_bindings_and_frozen_runtime_are_unchanged(module):
    inner = module.verify_frozen_bindings()
    assert inner.runtime_config()["execution_enabled"] is False
    assert inner.PREPARED_PLAN_SHA256 == module.PREPARED_SHA
    assert inner.PROPOSAL_SHA256 == module.RESOURCE_SHA
