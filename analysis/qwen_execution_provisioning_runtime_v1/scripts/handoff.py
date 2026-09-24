"""Co-located handoff to the unchanged inner supervisor; no remote bootstrap.

The provider transport is not an SSH deployer. These files, the pinned cache, the
original pod-local watchdog, and its runpodctl preflight must already be on the
approved pod. An external API shutdown worker cannot replace that inner watchdog.
"""
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import controller as outer
from live_policy import require_live, verify_source_bundle


class FrozenInnerControl:
    synthetic_only = False

    def __init__(self, preflight, authorization, cli_style, cli_executable):
        self.preflight, self.authorization = Path(preflight), Path(authorization)
        self.style, self.executable = cli_style, str(cli_executable)

    def verify_stopped_preflight(self, grant, *, current):
        # Check private bytes first, then call the reviewed verifier unchanged.
        outer.private_json(self.preflight)
        if outer.private_json(self.authorization) != grant:
            raise PermissionError("Inner authorization changed before preflight")
        if self.style not in {"modern", "legacy"} or Path(self.executable).name != "runpodctl":
            raise PermissionError("Exact reviewed runpodctl control path required")
        inner = outer.verify_frozen_bindings()
        watchdog = inner.load_module("outer_frozen_watchdog", inner.QWEN_EXECUTION / "scripts/stop_watchdog.py")
        watchdog.verify_stop_preflight(self.preflight, grant["pod_id"], self.style,
                                       self.executable, current=current)
        return {"pod_id": grant["pod_id"], "verified": True,
                "authorization_sha256": outer.sha(self.authorization)}


class InnerRunnerHandoff:
    synthetic_only = False

    def __init__(self, *, prepared, cache, output, authorization, watchdog_receipt,
                 stop_preflight, provider_observation):
        self.paths = {name: Path(value) for name, value in locals().copy().items()
                      if name != "self"}

    def frozen_gate_ready(self):
        inner = outer.verify_frozen_bindings()
        if inner.runtime_config().get("execution_enabled") is not True:
            raise PermissionError("Frozen inner execution remains disabled; no paid resume")
        return inner

    def validate_before_resume(self, controller, grant):
        # Do not read cache/model or start paid resume when the existing gate is off.
        inner = self.frozen_gate_ready()
        require_live("execution_enabled")
        verify_source_bundle(controller.grant["live_source_manifest_sha256"])
        if os.environ.get("RUNPOD_POD_ID") != grant["pod_id"]:
            raise PermissionError("Inner handoff must run on the exact approved pod")
        for name in ("authorization", "watchdog_receipt", "stop_preflight", "provider_observation"):
            outer.private_json(self.paths[name])
        if outer.private_json(self.paths["authorization"]) != grant:
            raise PermissionError("Handoff authorization differs from resume grant")
        if self.paths["output"].exists():
            raise FileExistsError("No inference output overwrite or continuation")
        inner.check_plan(self.paths["prepared"], outer.PREPARED_SHA)
        # This is the original live gate: same original pod-local watchdog and
        # receipt checks. We do not adapt an external-worker receipt to satisfy it.
        inner.live_gate(inner.ROOT / "configs/execution_plan.json", outer.EXECUTION_SHA,
            self.paths["authorization"], self.paths["watchdog_receipt"],
            self.paths["stop_preflight"], self.paths["provider_observation"])

    def run(self, controller, grant):
        self.validate_before_resume(controller, grant)
        controller.before_start()
        controller.verify_guard()
        if (grant["provider_deadline"] != controller.intent["outer_provider_deadline"]
                or grant["watchdog_stop_at"] != controller.intent["outer_watchdog_stop_at"]
                or outer.money(grant["maximum_usd"]) + controller.accrued_upper
                    > outer.money(controller.intent["maximum_usd"])):
            raise PermissionError("Handoff would reset the outer budget or deadline")
        controller.claim_operation("handoff")
        controller.event("inner-handoff-attempt", {
            "execution_plan_sha256": outer.EXECUTION_SHA,
            "outer_intent_sha256": controller.intent_sha,
            "authorization_sha256": outer.sha(self.paths["authorization"]),
            "provider_deadline": grant["provider_deadline"],
            "watchdog_stop_at": grant["watchdog_stop_at"]})
        inner = outer.verify_frozen_bindings()
        # The original launcher owns parent attestation, hard process-group timeout,
        # generation ledgers, cache/token parity, watchdog and independent decoding.
        ok = inner.launch_supervised(self.paths["prepared"], outer.PREPARED_SHA,
            inner.ROOT / "configs/execution_plan.json", outer.EXECUTION_SHA,
            self.paths["authorization"], self.paths["watchdog_receipt"],
            self.paths["stop_preflight"], self.paths["provider_observation"],
            self.paths["cache"], self.paths["output"])
        if not ok:
            raise outer.SafetyFailure("Inner runner failed; no retry or continuation")
        return True
