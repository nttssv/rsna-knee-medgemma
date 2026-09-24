import importlib.util
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qwen_live_test", ROOT / "scripts/qwen_live.py")
live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live)


class Tokenizer:
    def decode(self, ids, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return {tuple([11, 151645]): "output<eos>", (11,): "output"}[tuple(ids)]


def test_plan_has_fixed_20_generation_scope_and_pins():
    plan = live.execution_plan()
    plan_path = ROOT / "configs/execution_plan.json"
    assert live.validate_execution_plan(plan_path, live.sha(plan_path)) == plan
    assert plan["prepared_plan_sha256"] == live.PREPARED_PLAN_SHA256
    assert plan["resource_proposal_sha256"] == live.PROPOSAL_SHA256
    assert plan["maximum_generations"] == 20
    assert plan["run_order"] == ["control-1", "candidate-1", "candidate-2", "control-2"]
    assert plan["reports_per_run"] == 5
    assert all(plan[k] is False for k in ("training", "validation_inference", "full_development_inference",
        "bulk_extraction", "retry_or_repair", "real_execution_enabled_by_default"))


def test_default_live_gate_fails_before_reading_approval_or_cache(tmp_path):
    with pytest.raises(PermissionError, match="disabled"):
        live.live_gate(tmp_path / "missing-plan.json", "0" * 64,
            tmp_path / "missing-approval.json", tmp_path / "missing-watchdog.json",
            tmp_path / "missing-preflight.json")


def test_approval_exact_scope_is_accepted_only_as_synthetic_fixture(tmp_path):
    approval_time = "2026-09-24T12:50:00+00:00"
    start = "2026-09-24T12:55:00+00:00"
    deadline = "2026-09-24T13:55:00+00:00"
    grant = dict(approved=True, approval_reference="SYNTHETIC TEST ONLY",
        plan_sha256="a" * 64, prepared_plan_sha256=live.PREPARED_PLAN_SHA256,
        proposal_sha256=live.PROPOSAL_SHA256, approved_at=approval_time,
        provider_start_requested_at=start, provider_deadline=deadline,
        watchdog_stop_at="2026-09-24T13:50:00+00:00", maximum_usd=1.5,
        actual_compute_usd_per_hour=0.84, actual_storage_usd_per_hour=0.012,
        pod_id="synthetic-pod-fixture", gpu_name="NVIDIA RTX 6000 Ada Generation", gpu_count=1,
        cloud_type="SECURE", container_disk_gb=80, persistent_volume_gb=0,
        network_volume_id=None)
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(grant))
    path.chmod(0o600)
    now = datetime(2026, 9, 24, 13, 0, tzinfo=timezone.utc)
    assert live.validate_user_approval(path, "a" * 64, current=now)["pod_id"] == "synthetic-pod-fixture"


@pytest.mark.parametrize("key,value", [
    ("plan_sha256", "b" * 64), ("prepared_plan_sha256", "c" * 64),
    ("proposal_sha256", "d" * 64), ("gpu_count", 2), ("cloud_type", "COMMUNITY"),
    ("container_disk_gb", 81), ("persistent_volume_gb", 1), ("network_volume_id", "volume-x"),
    ("actual_compute_usd_per_hour", 0.85), ("actual_storage_usd_per_hour", 0.013),
    ("maximum_usd", 1.51), ("watchdog_stop_at", "2026-09-24T13:49:59+00:00"),
    ("provider_deadline", "2026-09-24T14:00:00+00:00"),
])
def test_approval_rejects_scope_or_deadline_drift(tmp_path, key, value):
    grant = dict(approved=True, approval_reference="SYNTHETIC TEST ONLY",
        plan_sha256="a" * 64, prepared_plan_sha256=live.PREPARED_PLAN_SHA256,
        proposal_sha256=live.PROPOSAL_SHA256, approved_at="2026-09-24T12:50:00+00:00",
        provider_start_requested_at="2026-09-24T12:55:00+00:00",
        provider_deadline="2026-09-24T13:55:00+00:00", watchdog_stop_at="2026-09-24T13:50:00+00:00",
        maximum_usd=1.5, actual_compute_usd_per_hour=0.84, actual_storage_usd_per_hour=0.012,
        pod_id="synthetic-pod-fixture", gpu_name="NVIDIA RTX 6000 Ada Generation", gpu_count=1,
        cloud_type="SECURE", container_disk_gb=80, persistent_volume_gb=0, network_volume_id=None)
    grant[key] = value
    if key == "provider_deadline":
        grant["watchdog_stop_at"] = "2026-09-24T13:55:00+00:00"
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(grant)); path.chmod(0o600)
    with pytest.raises(PermissionError):
        live.validate_user_approval(path, "a" * 64,
            current=datetime(2026, 9, 24, 13, 0, tzinfo=timezone.utc))


def test_approval_rejects_group_or_world_readable(tmp_path):
    path = tmp_path / "fixture.json"
    path.write_text("{}")
    path.chmod(0o644)
    with pytest.raises(PermissionError, match="0600"):
        live.validate_user_approval(path, "a" * 64)


def test_hardware_preflight_accepts_only_exact_pinned_runtime():
    name = "NVIDIA RTX 6000 Ada Generation"
    live.verify_reported_hardware([name], 1, "12.8", "2.8.0+cu128", True, True, 36.0)
    cases = [([name, name], 2, "12.8", "2.8.0+cu128", True, True, 48.0),
        (["Other GPU"], 1, "12.8", "2.8.0+cu128", True, True, 48.0),
        ([name], 1, "12.7", "2.8.0+cu128", True, True, 48.0),
        ([name], 1, "12.8", "2.8.0", True, True, 48.0),
        ([name], 1, "12.8", "2.8.0+cu128", False, True, 48.0),
        ([name], 1, "12.8", "2.8.0+cu128", True, False, 48.0),
        ([name], 1, "12.8", "2.8.0+cu128", True, True, 35.99)]
    for args in cases:
        with pytest.raises(RuntimeError):
            live.verify_reported_hardware(*args)


def test_independent_decode_requires_byte_exact_raw_and_full_text():
    result = dict(output_token_ids=[11, 151645], raw_output="output", decoded_with_special_tokens="output<eos>")
    assert live.independent_decode(Tokenizer(), result) == dict(raw_output="output", decoded_with_special_tokens="output<eos>")
    result["raw_output"] = "output "
    with pytest.raises(ValueError, match="raw output"):
        live.independent_decode(Tokenizer(), result)


def test_independent_decode_rejects_full_decode_mismatch():
    result = dict(output_token_ids=[11, 151645], raw_output="output", decoded_with_special_tokens="output")
    with pytest.raises(ValueError, match="special-token output"):
        live.independent_decode(Tokenizer(), result)


def test_incomplete_generation_stops_before_any_parser():
    class Backend:
        class E:
            tokenizer = Tokenizer()
        encoder = E()
    encoded = type("Encoded", (), {"input_tokens": 1})()
    with pytest.raises(RuntimeError, match="no retry"):
        live.validate_generation(dict(generation_status="oom", output_token_ids=[], output_tokens=0,
            raw_output="", decoded_with_special_tokens="", input_tokens=1), Backend(), encoded, "fixture")


def test_durable_generation_persists_attempt_and_raw_before_parse(monkeypatch, tmp_path):
    order = []
    paths = {}
    for label in ("attempt", "raw", "parsed"):
        path = tmp_path / (label + ".jsonl")
        paths[label] = path.open("w+")
    fds = {stream.fileno(): label for label, stream in paths.items()}
    real_fsync = live.os.fsync
    def tracked_fsync(fd):
        order.append("durable-" + fds[fd])
        return real_fsync(fd)
    monkeypatch.setattr(live.os, "fsync", tracked_fsync)
    monkeypatch.setattr(live, "validate_generation", lambda *args: ("primary", "secondary"))
    class Backend:
        def generate(self, encoded):
            order.append("generate")
            return {"generation_status": "completed", "output_token_ids": [2]}
    row = {"StudyInstanceUID": "synthetic-1", "report_sha256": "x", "Report": "synthetic"}
    live.durable_generation(Backend(), type("Encoded", (), {"rendered_prompt": "p", "input_ids": [1], "input_tokens": 1})(),
        row, "control-1", 0, 1, "synthetic-pod", paths["attempt"], paths["raw"], paths["parsed"])
    assert order == ["durable-attempt", "generate", "durable-raw", "durable-parsed"]
    for stream in paths.values():
        stream.close()


def test_expired_quote_and_stop_reserve_fail_closed(tmp_path):
    grant = dict(approved=True, approval_reference="SYNTHETIC TEST ONLY",
        plan_sha256="a" * 64, prepared_plan_sha256=live.PREPARED_PLAN_SHA256,
        proposal_sha256=live.PROPOSAL_SHA256, approved_at="2026-09-24T12:50:00+00:00",
        provider_start_requested_at="2026-09-24T12:55:00+00:00",
        provider_deadline="2026-09-24T13:55:00+00:00", watchdog_stop_at="2026-09-24T13:50:00+00:00",
        maximum_usd=1.5, actual_compute_usd_per_hour=0.84, actual_storage_usd_per_hour=0.012,
        pod_id="synthetic-pod-fixture", gpu_name="NVIDIA RTX 6000 Ada Generation", gpu_count=1,
        cloud_type="SECURE", container_disk_gb=80, persistent_volume_gb=0, network_volume_id=None)
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(grant)); path.chmod(0o600)
    with pytest.raises(PermissionError, match="quote expired"):
        live.validate_user_approval(path, "a" * 64,
            current=datetime(2026, 9, 26, 13, 0, tzinfo=timezone.utc))
    with pytest.raises(TimeoutError, match="watchdog stop reserve"):
        live.check_time_budget(grant, started=0, current=datetime.fromisoformat(grant["watchdog_stop_at"]))


def test_incomplete_generation_keeps_attempt_and_raw_before_failure(tmp_path):
    class Backend:
        def generate(self, encoded):
            return dict(generation_status="oom", output_token_ids=[], output_tokens=0,
                input_tokens=1, raw_output="", decoded_with_special_tokens="", runtime_seconds=1.0)
    encoded = type("Encoded", (), {"rendered_prompt": "p", "input_ids": [1], "input_tokens": 1})()
    row = {"StudyInstanceUID": "synthetic-1", "report_sha256": "x", "Report": "synthetic"}
    files = [ (tmp_path / name).open("w+") for name in ("attempts", "raw", "parsed") ]
    with pytest.raises(RuntimeError, match="no retry"):
        live.durable_generation(Backend(), encoded, row, "control-1", 0, 1, "synthetic-pod", *files)
    for file in files: file.flush()
    assert len((tmp_path / "attempts").read_text().splitlines()) == 1
    raw = json.loads((tmp_path / "raw").read_text())
    assert raw["generation"]["generation_status"] == "oom"
    assert (tmp_path / "parsed").read_text() == ""
    for file in files: file.close()


def test_complete_generation_stops_on_independent_token_raw_mismatch(monkeypatch):
    class Backend:
        class E:
            tokenizer = Tokenizer()
        encoder = E()
    encoded = type("Encoded", (), {"input_tokens": 1})()
    result = dict(generation_status="completed", output_token_ids=[11, 151645],
        output_tokens=2, input_tokens=1, runtime_seconds=1.0,
        raw_output="wrong", decoded_with_special_tokens="output<eos>")
    with pytest.raises(ValueError, match="raw output"):
        live.validate_generation(result, Backend(), encoded, "fixture")


def test_generation_is_not_started_without_330_second_stop_margin():
    grant = {"watchdog_stop_at": "2026-09-24T13:50:00+00:00"}
    live.check_time_budget(grant, time.monotonic(),
        current=datetime(2026, 9, 24, 13, 44, 29, tzinfo=timezone.utc),
        minimum_remaining_seconds=330)
    with pytest.raises(TimeoutError):
        live.check_time_budget(grant, time.monotonic(),
            current=datetime(2026, 9, 24, 13, 44, 30, tzinfo=timezone.utc),
            minimum_remaining_seconds=330)


def test_watchdog_liveness_failure_blocks_next_generation(monkeypatch):
    calls = []
    def dead_watchdog(*args):
        calls.append("checked")
        raise PermissionError("watchdog process identity is unverified")
    monkeypatch.setattr(live, "verify_live_watchdog", dead_watchdog)
    current = datetime.now(timezone.utc)
    with pytest.raises(PermissionError, match="watchdog process identity"):
        live.pre_generation_gate("watch-receipt", {"watchdog_stop_at": (current.replace(microsecond=0) +
            timedelta(seconds=600)).isoformat()}, "stop-preflight",
            time.monotonic(), current=current)
    assert calls == ["checked"]


def test_supervisor_hard_timeout_kills_and_reaps_process_group():
    import sys
    ok, reason = live.supervise([sys.executable, "-c", "import time; time.sleep(30)"], 0.15)
    assert ok is False
    assert reason == "hard_timeout"


def test_worker_requires_parent_pid_nonce_and_bound_receipts(tmp_path):
    output = tmp_path / "run"
    output.mkdir(mode=0o700)
    paths = {}
    for name in ("authorization", "watchdog", "preflight"):
        path = tmp_path / (name + ".json")
        path.write_text("{}")
        paths[name] = path
    nonce = "synthetic-nonce"
    start = dict(parent_pid=os.getppid(), execution_plan_sha256="e" * 64,
        prepared_plan_sha256=live.PREPARED_PLAN_SHA256,
        authorization_sha256=live.sha(paths["authorization"]),
        watchdog_receipt_sha256=live.sha(paths["watchdog"]),
        stop_preflight_sha256=live.sha(paths["preflight"]))
    live.write_new(output / "supervisor_start.json", start)
    live.write_new(output / "supervisor_dispatch.json", dict(
        supervisor_start_sha256=live.sha(output / "supervisor_start.json"),
        nonce_sha256=live.digest(nonce)))
    read_fd, write_fd = os.pipe()
    with os.fdopen(write_fd, "w") as stream:
        json.dump(dict(nonce=nonce, parent_pid=os.getppid(),
            execution_plan_sha256="e" * 64,
            prepared_plan_sha256=live.PREPARED_PLAN_SHA256), stream)
    context = live.verify_worker_attestation(read_fd, output, "e" * 64,
        live.PREPARED_PLAN_SHA256, paths["authorization"], paths["watchdog"], paths["preflight"])
    assert context["parent_pid"] == os.getppid()


def test_worker_rejects_wrong_pipe_nonce(tmp_path):
    output = tmp_path / "run"
    output.mkdir(mode=0o700)
    paths = {}
    for name in ("authorization", "watchdog", "preflight"):
        path = tmp_path / (name + ".json")
        path.write_text("{}")
        paths[name] = path
    start = dict(parent_pid=os.getppid(), execution_plan_sha256="e" * 64,
        prepared_plan_sha256=live.PREPARED_PLAN_SHA256,
        authorization_sha256=live.sha(paths["authorization"]),
        watchdog_receipt_sha256=live.sha(paths["watchdog"]),
        stop_preflight_sha256=live.sha(paths["preflight"]))
    live.write_new(output / "supervisor_start.json", start)
    live.write_new(output / "supervisor_dispatch.json", dict(
        supervisor_start_sha256=live.sha(output / "supervisor_start.json"),
        nonce_sha256=live.digest("expected")))
    read_fd, write_fd = os.pipe()
    with os.fdopen(write_fd, "w") as stream:
        json.dump(dict(nonce="wrong", parent_pid=os.getppid(),
            execution_plan_sha256="e" * 64,
            prepared_plan_sha256=live.PREPARED_PLAN_SHA256), stream)
    with pytest.raises(PermissionError, match="supervisor provenance"):
        live.verify_worker_attestation(read_fd, output, "e" * 64,
            live.PREPARED_PLAN_SHA256, paths["authorization"], paths["watchdog"], paths["preflight"])


def test_supervisor_deadline_is_minimum_of_session_and_stop_bounds():
    now = datetime.now(timezone.utc)
    grant = {"watchdog_stop_at": (now + timedelta(seconds=5000)).isoformat()}
    assert live.bounded_supervisor_timeout(grant, current=now) == 1799
    grant["watchdog_stop_at"] = (now + timedelta(seconds=1000)).isoformat()
    assert live.bounded_supervisor_timeout(grant, current=now) == 999
    grant["watchdog_stop_at"] = (now + timedelta(seconds=1)).isoformat()
    with pytest.raises(TimeoutError):
        live.bounded_supervisor_timeout(grant, current=now)


def test_launcher_uses_execute_flag_and_persists_supervisor_result(monkeypatch, tmp_path):
    now = datetime.now(timezone.utc)
    grant = {"pod_id": "synthetic-pod", "provider_deadline": (now + timedelta(seconds=3900)).isoformat(),
        "watchdog_stop_at": (now + timedelta(seconds=3600)).isoformat()}
    monkeypatch.setattr(live, "validate_execution_plan", lambda *args: {})
    monkeypatch.setattr(live, "check_plan", lambda *args: (None, []))
    monkeypatch.setattr(live, "live_gate", lambda *args, **kwargs: grant)
    inputs = []
    for name in ("authorization", "watchdog", "preflight"):
        path = tmp_path / name
        path.write_text("synthetic")
        inputs.append(path)
    output = tmp_path / "execution-output"
    observed = {}
    def fake_supervise(command, timeout, pass_fds=()):
        observed["command"] = command
        observed["timeout"] = timeout
        assert len(pass_fds) == 1
        result = dict(status="completed", execution_plan_sha256="e" * 64,
            prepared_plan_sha256=live.PREPARED_PLAN_SHA256, generations=20,
            run_order=live.execution_plan()["run_order"])
        live.write_new(output / "session_result.json", result)
        return True, "completed"
    monkeypatch.setattr(live, "supervise", fake_supervise)
    ok = live.launch_supervised(tmp_path / "prepared", live.PREPARED_PLAN_SHA256,
        ROOT / "configs/execution_plan.json", "e" * 64, *inputs[:1], inputs[1], inputs[2],
        tmp_path / "cache", output)
    assert ok is True
    assert "--execute" in observed["command"]
    assert "--_worker" in observed["command"]
    assert observed["timeout"] == 1799
    assert json.loads((output / "supervisor_result.json").read_text())["status"] == "completed"


def test_launcher_records_hard_timeout_and_partial_hashes(monkeypatch, tmp_path):
    now = datetime.now(timezone.utc)
    grant = {"pod_id": "synthetic-pod", "provider_deadline": (now + timedelta(seconds=3900)).isoformat(),
        "watchdog_stop_at": (now + timedelta(seconds=3600)).isoformat()}
    monkeypatch.setattr(live, "validate_execution_plan", lambda *args: {})
    monkeypatch.setattr(live, "check_plan", lambda *args: (None, []))
    monkeypatch.setattr(live, "live_gate", lambda *args, **kwargs: grant)
    paths = []
    for name in ("authorization", "watchdog", "preflight"):
        path = tmp_path / name; path.write_text("synthetic"); paths.append(path)
    output = tmp_path / "timed-out-output"
    def fake_timeout(command, timeout, pass_fds=()):
        (output / "control-1").mkdir()
        partial = output / "control-1/raw.jsonl"
        partial.write_text("durable partial receipt\n")
        return False, "hard_timeout"
    monkeypatch.setattr(live, "supervise", fake_timeout)
    ok = live.launch_supervised(tmp_path / "prepared", live.PREPARED_PLAN_SHA256,
        ROOT / "configs/execution_plan.json", "f" * 64, paths[0], paths[1], paths[2],
        tmp_path / "cache", output)
    assert ok is False
    receipt = json.loads((output / "supervisor_result.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["process_status"] == "hard_timeout"
    assert receipt["partial_artifacts_sha256"]["control-1/raw.jsonl"] == live.sha(output / "control-1/raw.jsonl")
    assert "session_result.json" not in receipt["partial_artifacts_sha256"]
