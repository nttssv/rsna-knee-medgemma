import importlib.util
import json
import os
import time
from datetime import datetime, timezone
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
