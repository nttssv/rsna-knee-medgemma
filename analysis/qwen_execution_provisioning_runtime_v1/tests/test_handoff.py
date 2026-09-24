"""Exact inner-supervisor forwarding with fabricated private paths; no model use."""
from decimal import Decimal
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import handoff as h


def fixture(tmp_path, monkeypatch, *, completed=True):
    paths = {key: tmp_path / key for key in ("prepared", "cache", "output", "authorization",
             "watchdog_receipt", "stop_preflight", "provider_observation")}
    paths["authorization"].write_text("synthetic authorization fixture")
    runner = h.InnerRunnerHandoff(**paths)
    calls = []
    inner = SimpleNamespace(ROOT=tmp_path / "frozen-inner",
        launch_supervised=lambda *args: calls.append(("inner", args)) or completed)
    monkeypatch.setattr(runner, "validate_before_resume", lambda *args: calls.append(("validate",)))
    monkeypatch.setattr(h.outer, "verify_frozen_bindings", lambda: inner)
    grant = {"provider_deadline": "2026-09-24T19:00:00+00:00",
             "watchdog_stop_at": "2026-09-24T18:55:00+00:00", "maximum_usd": 1.40}
    controller = SimpleNamespace(
        intent={"outer_provider_deadline": grant["provider_deadline"],
                "outer_watchdog_stop_at": grant["watchdog_stop_at"], "maximum_usd": 1.50},
        intent_sha="synthetic-intent", accrued_upper=Decimal("0.10"),
        before_start=lambda: calls.append(("clock",)),
        verify_guard=lambda: calls.append(("guard",)),
        claim_operation=lambda value: calls.append(("claim", value)),
        event=lambda *args: calls.append(("receipt", args)))
    return runner, controller, grant, calls, paths, inner


def test_handoff_calls_exact_frozen_launcher_after_claim_and_receipt(tmp_path, monkeypatch):
    runner, ctl, grant, calls, p, inner = fixture(tmp_path, monkeypatch)
    assert runner.run(ctl, grant) is True
    assert [item[0] for item in calls] == ["validate", "clock", "guard", "claim", "receipt", "inner"]
    assert calls[-1][1] == (p["prepared"], h.outer.PREPARED_SHA,
        inner.ROOT / "configs/execution_plan.json", h.outer.EXECUTION_SHA,
        p["authorization"], p["watchdog_receipt"], p["stop_preflight"],
        p["provider_observation"], p["cache"], p["output"])


@pytest.mark.parametrize("drift", ["deadline", "stop", "cost"])
def test_handoff_rejects_reset_before_claim_or_launch(tmp_path, monkeypatch, drift):
    runner, ctl, grant, calls, *_ = fixture(tmp_path, monkeypatch)
    if drift == "deadline":
        grant["provider_deadline"] = "2026-09-24T20:00:00+00:00"
    elif drift == "stop":
        grant["watchdog_stop_at"] = "2026-09-24T19:55:00+00:00"
    else:
        ctl.accrued_upper = Decimal("0.100000001")
    with pytest.raises(PermissionError):
        runner.run(ctl, grant)
    assert not any(call[0] in {"claim", "inner"} for call in calls)


def test_incomplete_inner_run_is_terminal_no_repeat(tmp_path, monkeypatch):
    runner, ctl, grant, calls, *_ = fixture(tmp_path, monkeypatch, completed=False)
    with pytest.raises(h.outer.SafetyFailure, match="no retry"):
        runner.run(ctl, grant)
    assert sum(call[0] == "inner" for call in calls) == 1
