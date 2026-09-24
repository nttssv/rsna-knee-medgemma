"""Bounded provisioning lifecycle with source-bound, disabled production adapters.

Synthetic interfaces exercise decisions without network access. Production transport
uses an external process-group deadline; the detached guard survives allocator exit.
All committed live switches remain false, including the unchanged inner runner.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Protocol

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT.parent
PROVISIONING_SHA = "0770846122bb4bb4953c7e728afa1c254ae569c8f756b3e905e6c095a2ed2842"
EXECUTION_SHA = "ab82f51cf3883e93b2c33bf750a32148297226ebd3e9ada0febb54d18d85a4f5"
PREPARED_SHA = "c4586bb0c65163244dac80344ea96f691493a1b89a5911eefcba54f2384c74fc"
RESOURCE_SHA = "884dc38d2004abb44c64d2363e525969bf7f7fabd35f86eb5d03144bf34e72e2"
IMAGE = "runpod/pytorch@sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851"
STARTUP = ["/bin/sleep", "infinity"]
FIXED = {
    "cloud_type": "SECURE", "region": "CA-MTL-1", "gpu_name": "NVIDIA A40",
    "gpu_count": 1, "gpu_vram_gb": 48, "container_disk_gb": 80,
    "persistent_volume_gb": 0, "network_volume_id": None,
    "image_ref": IMAGE, "startup_command": STARTUP, "template_id": None,
}
MAXIMUM = Decimal("1.50")
CEILING_PER_HOUR = Decimal("0.502")
STOPPED = {"STOPPED", "EXITED"}
sys.path.insert(0, str(Path(__file__).resolve().parent))


class SafetyFailure(RuntimeError):
    """The current attempt ends; no retry or alternate allocation is permitted."""


class Provider(Protocol):
    def find_by_name(self, name: str, *, timeout: float) -> list[dict]: ...
    def create(self, request: dict, *, timeout: float) -> str: ...
    def read(self, pod_id: str, *, timeout: float) -> dict: ...
    def stop(self, pod_id: str, *, timeout: float) -> None: ...
    def resume(self, pod_id: str, *, timeout: float) -> None: ...


class IndependentObserver(Protocol):
    """A separate read path, not the mutation response or its cached object."""
    def read(self, pod_id: str, *, timeout: float) -> dict: ...


class ExternalGuard(Protocol):
    """Future implementation must independently verify an already running worker.

    It must outlive this process, monitor its durable intent, reconcile a lost create
    response, and stop the one bound pod. A self-reported PID JSON is insufficient.
    verify() must test process/command/environment/receipt identity, not just parse it.
    """
    def verify(self, intent: dict, intent_sha256: str, *, current: datetime) -> dict: ...


class InnerControl(Protocol):
    """Future implementation calls the frozen exact-pod stopped preflight verifier."""
    def verify_stopped_preflight(self, grant: dict, *, current: datetime) -> dict: ...


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def accrued_ceiling(first: datetime, stopped: datetime) -> Decimal:
    elapsed = stopped - first
    if elapsed.total_seconds() < 0:
        raise SafetyFailure("Clock moved before the first create timestamp")
    seconds = Decimal(elapsed.days * 86400 + elapsed.seconds) + Decimal(elapsed.microseconds) / 1000000
    # Round upward to a nano-dollar, never understate a repeating decimal fraction.
    return (CEILING_PER_HOUR * seconds / 3600).quantize(Decimal("0.000000001"), rounding=ROUND_CEILING)


def time_value(value: str) -> datetime:
    if not isinstance(value, str):
        raise SafetyFailure("Timezone-aware timestamp required")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SafetyFailure("Invalid timestamp") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise SafetyFailure("Timezone-aware timestamp required")
    return result.astimezone(timezone.utc)


def money(value) -> Decimal:
    if type(value) not in (int, float):
        raise SafetyFailure("Finite nonnegative numeric amount required")
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise SafetyFailure("Finite nonnegative numeric amount required")
    return result


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SafetyFailure("Duplicate JSON key")
        result[key] = value
    return result


def private_json(path: Path) -> dict:
    path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != os.getuid()):
            raise SafetyFailure("Private owner-only 0600 regular file required")
        with os.fdopen(fd, "r", closefd=False) as stream:
            result = json.load(stream, object_pairs_hook=no_duplicates)
    finally:
        os.close(fd)
    if not isinstance(result, dict):
        raise SafetyFailure("JSON object required")
    return result


def write_exclusive(path: Path, value: dict) -> str:
    """Durable, exclusive, private record. Never truncate or replace a receipt."""
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(fd)
    finally:
        os.close(fd)
    parent = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    return hashlib.sha256(payload).hexdigest()


def frozen_runtime():
    module_path = ANALYSIS / "qwen_execution_runtime_alt_v1/scripts/qwen_live.py"
    spec = importlib.util.spec_from_file_location("provisioning_frozen_inner", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_frozen_bindings():
    paths = {
        ANALYSIS / "qwen_execution_provisioning_v1/configs/provisioning_proposal.json": PROVISIONING_SHA,
        ANALYSIS / "qwen_execution_runtime_alt_v1/configs/execution_plan.json": EXECUTION_SHA,
        ANALYSIS / "qwen_execution_resource_alt_v1/configs/resource_proposal.json": RESOURCE_SHA,
    }
    for path, expected in paths.items():
        if sha(path) != expected:
            raise SafetyFailure("Frozen source/proposal binding differs")
    inner = frozen_runtime()
    inner.validate_execution_plan(ANALYSIS / "qwen_execution_runtime_alt_v1/configs/execution_plan.json",
                                  EXECUTION_SHA)
    return inner


def validate_outer_approval(grant: dict, now: datetime):
    """Validate a supplied record. This function never creates user authorization."""
    if grant.get("approved") is not True or not grant.get("approval_reference"):
        raise SafetyFailure("Explicit provisioning approval required")
    bindings = {
        "provisioning_proposal_sha256": PROVISIONING_SHA,
        "reviewed_execution_plan_sha256": EXECUTION_SHA,
        "prepared_plan_sha256": PREPARED_SHA,
        "alternate_resource_proposal_sha256": RESOURCE_SHA,
        "controller_source_sha256": sha(Path(__file__)),
    }
    for key, expected in bindings.items():
        if grant.get(key) != expected:
            raise SafetyFailure(f"Provisioning approval binding differs: {key}")
    name = grant.get("intent_name", "")
    if not re.fullmatch(r"qwen-provision-[0-9a-f]{32}", name):
        raise SafetyFailure("One high-entropy exact intent name required")
    for key, expected in FIXED.items():
        if key not in grant or grant[key] != expected or type(grant[key]) is not type(expected):
            raise SafetyFailure(f"Exact provisioning resource differs: {key}")
    if grant.get("startup_command_sha256") != digest(STARTUP):
        raise SafetyFailure("Startup command hash differs")
    compute = money(grant.get("actual_compute_usd_per_hour"))
    storage = money(grant.get("actual_storage_usd_per_hour"))
    maximum = money(grant.get("maximum_usd"))
    if not 0 < compute <= Decimal("0.49") or storage > Decimal("0.012"):
        raise SafetyFailure("Provisioning rate ceiling exceeded")
    if not 0 < maximum <= MAXIMUM:
        raise SafetyFailure("Provisioning budget ceiling exceeded")
    seconds = grant.get("maximum_provider_seconds")
    if type(seconds) is not int or not 300 < seconds <= 3600:
        raise SafetyFailure("Invalid outer provider window")
    if CEILING_PER_HOUR * Decimal(seconds) / 3600 > maximum:
        raise SafetyFailure("Approved budget cannot cover the bounded provider window")
    approved = time_value(grant.get("approved_at"))
    observed = time_value(grant.get("quote_observed_at"))
    expires = time_value(grant.get("quote_valid_until"))
    if not approved <= observed <= now < expires or (now - observed).total_seconds() > 600:
        raise SafetyFailure("Fresh post-approval selected-allocation quote required")


class LocalController:
    """Lifecycle core; production assembly additionally enforces disabled live gates.

    State cannot be reloaded to retry interrupted create/resume. Independent durable
    operation claims prevent replay through a new output directory or process.
    """
    def __init__(self, output: Path, authorization: Path, provider: Provider,
                 observer: IndependentObserver, guard: ExternalGuard,
                 inner_control: InnerControl, clock, *, _allow_live=False, handoff=None):
        modes = [getattr(adapter, "synthetic_only", None)
                 for adapter in (provider, observer, guard, inner_control)]
        self.live = not all(mode is True for mode in modes)
        if self.live:
            if not _allow_live or not all(mode is False for mode in modes):
                raise SafetyFailure("Live adapters require the source-bound production factory")
            from live_policy import require_live
            require_live("provisioning_enabled")
            require_live("external_shutdown_enabled")
        if provider is observer:
            raise SafetyFailure("Stop verification requires an independent observer")
        self.inner = verify_frozen_bindings()
        self.clock = clock
        self.grant = private_json(authorization)
        validate_outer_approval(self.grant, self.now())
        self.authorization_sha = sha(authorization)
        self.authorization_path = Path(authorization)
        self.once_ledger = None
        self.handoff = handoff
        if self.live:
            from live_policy import OnceLedger, verify_source_bundle
            verify_source_bundle(self.grant.get("live_source_manifest_sha256", ""))
            self.once_ledger = OnceLedger(Path(self.grant["operation_ledger_dir"]),
                self.grant["intent_name"], {"authorization_sha256": self.authorization_sha,
                "live_source_manifest_sha256": self.grant["live_source_manifest_sha256"]})
        self.provider, self.observer, self.guard = provider, observer, guard
        self.inner_control = inner_control
        self.output = Path(output)
        self.output.mkdir(mode=0o700)  # Existing directory is terminal, never resume it.
        self.events = 0
        self.phase = "PREPARED"
        self.intent = None
        self.intent_sha = None
        self.pod_id = None
        self.machine_id = None
        self.stopped_at = None
        self.accrued_upper = Decimal("0")
        self.cost_bound_verified = False
        self.shutdown_receipts_durable = True
        self._create_attempted = False
        self._resume_attempted = False
        self.event("prepared", {"authorization_sha256": self.authorization_sha,
                                "controller_source_sha256": sha(Path(__file__))})

    def now(self):
        return time_value(self.clock().isoformat())

    def claim_operation(self, operation):
        if self.once_ledger is not None:
            self.once_ledger.claim(operation)

    def event(self, kind, fields=None):
        self.events += 1
        return write_exclusive(self.output / f"{self.events:03d}-{kind}.json",
            {"kind": kind, "recorded_at": self.now().isoformat(), **(fields or {})})

    def timeout(self, limit=30):
        remaining = (time_value(self.intent["outer_provider_deadline"]) - self.now()).total_seconds()
        if remaining <= 0:
            raise SafetyFailure("Outer provider deadline reached; external shutdown required")
        return min(float(limit), remaining)

    def before_start(self):
        if self.live:
            from live_policy import verify_source_bundle
            verify_source_bundle(self.grant["live_source_manifest_sha256"])
        if (sha(self.output / "intent.json") != self.intent_sha
                or private_json(self.output / "intent.json") != self.intent
                or sha(self.authorization_path) != self.authorization_sha
                or private_json(self.authorization_path) != self.grant):
            raise SafetyFailure("Immutable intent or authorization drift detected")
        first = time_value(self.intent["first_create_requested_at"])
        deadline = time_value(self.intent["outer_provider_deadline"])
        if (not first <= self.now() < deadline
                or (deadline - first).total_seconds() != self.grant["maximum_provider_seconds"]
                or time_value(self.intent["outer_watchdog_stop_at"]) != deadline - timedelta(seconds=300)
                or self.intent["maximum_usd"] != self.grant["maximum_usd"]):
            raise SafetyFailure("Immutable outer clock/budget differs")
        if self.now() >= time_value(self.intent["outer_watchdog_stop_at"]):
            raise SafetyFailure("Outer shutdown reserve reached")

    def shutdown_event(self, kind, fields=None):
        """Do not condition emergency stop on working disk; never hide lost receipts."""
        try:
            return self.event(kind, fields)
        except Exception:
            self.shutdown_receipts_durable = False
            return None

    def verify_guard(self, require_create_attempt=False):
        options = {"require_create_attempt": True} if require_create_attempt else {}
        receipt = self.guard.verify(dict(self.intent), self.intent_sha, current=self.now(), **options)
        required = {
            "intent_sha256": self.intent_sha, "intent_name": self.intent["intent_name"],
            "outer_provider_deadline": self.intent["outer_provider_deadline"],
            "outer_watchdog_stop_at": self.intent["outer_watchdog_stop_at"],
            "independent_of_allocating_process": True, "alive": True,
            "command_and_environment_verified": True, "stop_capable": True,
        }
        for key, value in required.items():
            if receipt.get(key) != value or type(receipt.get(key)) is not type(value):
                raise SafetyFailure(f"External shutdown guard verification failed: {key}")
        pid = receipt.get("pid")
        verified = time_value(receipt.get("verified_at"))
        if type(pid) is not int or pid <= 0 or pid == os.getpid():
            raise SafetyFailure("Shutdown guard must be a separate independently verified process")
        if not 0 <= (self.now() - verified).total_seconds() <= 5:
            raise SafetyFailure("Shutdown guard verification is stale")
        if require_create_attempt and receipt.get("durable_create_attempt_verified") is not True:
            raise SafetyFailure("Independent guard has not acknowledged durable creation ownership")
        self.event("external-guard-verified", receipt)

    def resource(self, observed, *, require_name=True):
        if not 0 <= (self.now() - time_value(observed.get("observed_at"))).total_seconds() <= 5:
            raise SafetyFailure("Provider allocation read is stale or in the future")
        for key, value in FIXED.items():
            if key not in observed or observed[key] != value or type(observed[key]) is not type(value):
                raise SafetyFailure(f"Provider resource differs: {key}")
        if require_name and observed.get("name") != self.grant["intent_name"]:
            raise SafetyFailure("Provider intent name differs")
        for key in ("actual_compute_usd_per_hour", "actual_storage_usd_per_hour"):
            if money(observed.get(key)) != money(self.grant[key]):
                raise SafetyFailure(f"Provider signed-in rate differs: {key}")
        if not isinstance(observed.get("machine_id"), str) or not observed["machine_id"]:
            raise SafetyFailure("Exact physical machine identity is required")
        if self.pod_id is not None and observed.get("pod_id") != self.pod_id:
            raise SafetyFailure("Provider pod identity differs")
        if self.machine_id is not None and observed["machine_id"] != self.machine_id:
            raise SafetyFailure("Replacement physical host is forbidden")

    def stop_exact(self):
        """At most three bounded stop attempts, each followed by an independent read.

        Live hard deadlines remain unimplemented: future transport must kill/reap
        hung calls externally. A mutation return never establishes stopped billing.
        """
        if self.pod_id is None:
            raise SafetyFailure("Cannot stop an unbound pod")
        for attempt in range(1, 4):
            try:
                timeout = self.timeout()
                self.shutdown_event("stop-attempt", {"pod_id": self.pod_id, "attempt": attempt,
                    "timeout_seconds": timeout})
                try:
                    self.provider.stop(self.pod_id, timeout=timeout)
                except Exception as exc:
                    self.shutdown_event("stop-response-unconfirmed", {"error_type": type(exc).__name__})
                observed = self.observer.read(self.pod_id, timeout=self.timeout())
                if observed.get("pod_id") != self.pod_id:
                    raise SafetyFailure("Independent observer returned a different pod")
                at = time_value(observed.get("observed_at"))
                if not 0 <= (self.now() - at).total_seconds() <= 5:
                    raise SafetyFailure("Independent stop read is stale or in the future")
                if observed.get("provider_state") not in STOPPED or money(observed.get("current_total_usd_per_hour")) != 0:
                    raise SafetyFailure("Stopped state and zero hourly charge remain unverified")
                self.stopped_at = self.now()
                # Includes ambiguous creates, reads, stopping, and any stopped waiting.
                if self.cost_bound_verified:
                    self.accrued_upper = max(self.accrued_upper,
                        accrued_ceiling(time_value(self.intent["first_create_requested_at"]), self.stopped_at))
                self.shutdown_event("independently-stopped", {"pod_id": self.pod_id,
                    "provider_observation": observed,
                    "accrued_upper_usd": str(self.accrued_upper) if self.cost_bound_verified else None,
                    "cost_bound_status": "CEILING_VERIFIED" if self.cost_bound_verified else "UNVERIFIED",
                    "outer_provider_deadline": self.intent["outer_provider_deadline"]})
                return observed
            except Exception as exc:
                self.shutdown_event("stop-unverified", {"attempt": attempt, "error_type": type(exc).__name__})
        raise SafetyFailure("Shutdown unverified after three attempts; external guard/operator required")

    def provision_once(self):
        if self.phase != "PREPARED" or self._create_attempted:
            raise SafetyFailure("No second creation, continuation, or replacement allocation")
        self.phase = "PROVISIONING"
        try:
            first = self.now()
            deadline = first + timedelta(seconds=self.grant["maximum_provider_seconds"])
            self.intent = {
                "intent_name": self.grant["intent_name"],
                "authorization_sha256": self.authorization_sha,
                "provisioning_proposal_sha256": PROVISIONING_SHA,
                "controller_source_sha256": sha(Path(__file__)),
                "first_create_requested_at": first.isoformat(),
                "outer_provider_deadline": deadline.isoformat(),
                "outer_watchdog_stop_at": (deadline - timedelta(seconds=300)).isoformat(),
                "maximum_usd": self.grant["maximum_usd"],
                "allocation": {**FIXED, "actual_compute_usd_per_hour": self.grant["actual_compute_usd_per_hour"],
                               "actual_storage_usd_per_hour": self.grant["actual_storage_usd_per_hour"]},
            }
            self.intent_sha = write_exclusive(self.output / "intent.json", self.intent)
            # The initial read is inside the same conservative immutable clock.
            # It must precede guard arming and any creation attempt.
            existing = self.provider.find_by_name(self.grant["intent_name"], timeout=self.timeout())
            self.event("intent-name-preread", {"matches": len(existing)})
            if existing:
                raise SafetyFailure("Intent name already exists; do not create or stop anything")
            self.verify_guard()
            self.before_start()
            # Recheck quote immediately before sole dispatch, including guard setup time.
            validate_outer_approval(self.grant, self.now())
            self.claim_operation("create")
            self.event("create-attempt", {"intent_sha256": self.intent_sha, "maximum_attempts": 1})
            self._create_attempted = True
            if self.live:
                # Survives allocator death combined with subsequent ledger I/O loss.
                # No request leaves until the independent worker caches ownership.
                self.verify_guard(require_create_attempt=True)
                self.before_start()
                validate_outer_approval(self.grant, self.now())
            try:
                pod = self.provider.create({**FIXED, "name": self.grant["intent_name"]}, timeout=self.timeout())
                if not isinstance(pod, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", pod):
                    raise SafetyFailure("Missing or invalid create response identity")
                self.pod_id = pod
                self.event("create-identity-returned", {"pod_id": pod})
                observed = self.provider.read(pod, timeout=self.timeout())
                self.resource(observed)
            except Exception as exc:
                self.event("create-response-unconfirmed", {"error_type": type(exc).__name__})
                if self.pod_id is not None:
                    raise
                # Exactly one read-only reconciliation; never another create request.
                candidates = self.provider.find_by_name(self.grant["intent_name"], timeout=self.timeout())
                self.event("create-reconciliation", {"matches": len(candidates)})
                if len(candidates) != 1:
                    raise SafetyFailure("Create identity unresolved; independent guard/operator required")
                observed = candidates[0]
                pod = observed.get("pod_id")
                if not isinstance(pod, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", pod):
                    raise SafetyFailure("Reconciled identity is invalid")
                if observed.get("name") != self.grant["intent_name"]:
                    raise SafetyFailure("Reconciliation did not return the exact unique intent")
                # Unique high-entropy name plus empty preread identifies the owned
                # cleanup target. It is not accepted for later use until all fields
                # match; any mismatch still stops this potentially billable pod.
                self.pod_id = pod
                self.event("reconciled-cleanup-target", {"pod_id": pod})
                self.resource(observed)
            self.machine_id = observed["machine_id"]
            self.cost_bound_verified = True
            self.event("allocation-bound", {"pod_id": self.pod_id, "machine_id": self.machine_id,
                "provider_observation": observed, "intent_sha256": self.intent_sha})
        except Exception:
            self.phase = "FAILED"
            raise
        finally:
            if self.pod_id is not None:
                try:
                    self.stop_exact()
                except Exception:
                    self.phase = "FAILED_SHUTDOWN_UNVERIFIED"
                    raise
        if not self.shutdown_receipts_durable:
            self.phase = "FAILED_RECEIPT_DURABILITY"
            raise SafetyFailure("Stopped pod verified but shutdown receipt durability failed")
        self.phase = "STOPPED_PREPARATION_COMPLETE"
        self.event("preparation-complete", {"inference_unlocked": False})
        return self.pod_id

    def resume_once(self, inner_authorization: Path, stopped_observation: Path):
        """One later resume, with unchanged inner gates and immediate final shutdown."""
        if self.phase != "STOPPED_PREPARATION_COMPLETE" or self._resume_attempted:
            raise SafetyFailure("One later resume only after verified stopped preparation")
        self.phase = "RESUME_VALIDATING"
        try:
            if self.live:
                from live_policy import require_live
                require_live("resume_enabled")
                if self.handoff is None:
                    raise SafetyFailure("A verified inner handoff is required before paid resume")
                self.handoff.frozen_gate_ready()
            self.before_start()
            now = self.now()
            grant = self.inner.validate_user_approval(inner_authorization, EXECUTION_SHA, current=now)
            observed = private_json(stopped_observation)
            self.inner.verify_provider_observation(grant, observed, current=now)
            for key in ("pod_id", "region", "gpu_name", "gpu_count", "cloud_type",
                        "container_disk_gb", "persistent_volume_gb", "network_volume_id",
                        "actual_compute_usd_per_hour", "actual_storage_usd_per_hour"):
                expected = self.pod_id if key == "pod_id" else self.grant[key]
                if grant[key] != expected or type(grant[key]) is not type(expected):
                    raise SafetyFailure(f"Inner grant differs from outer allocation: {key}")
            if (time_value(grant["provider_deadline"]) != time_value(self.intent["outer_provider_deadline"])
                    or time_value(grant["watchdog_stop_at"]) != time_value(self.intent["outer_watchdog_stop_at"])):
                raise SafetyFailure("Inner grant must retain the immutable outer deadline and reserve")
            start = time_value(grant["provider_start_requested_at"])
            at = time_value(observed["observed_at"])
            if not self.stopped_at <= time_value(grant["approved_at"]) <= at <= start <= now:
                raise SafetyFailure("Later approval/observation must follow true stopped verification")
            if (now - start).total_seconds() > 5 or (now - at).total_seconds() > 600:
                raise SafetyFailure("Fresh observation and current resume-request timestamp required")
            if money(grant["maximum_usd"]) + self.accrued_upper > money(self.intent["maximum_usd"]):
                raise SafetyFailure("Inner maximum plus accrued provisioning upper bound exceeds outer budget")
            if not self.cost_bound_verified or not self.shutdown_receipts_durable:
                raise SafetyFailure("Initial cost bound and durable shutdown evidence are required")
            if self.handoff is not None:
                self.handoff.validate_before_resume(self, grant)
            # Fresh independent live read supplements the frozen stopped snapshot.
            fresh = self.observer.read(self.pod_id, timeout=self.timeout())
            self.resource(fresh)
            if fresh.get("provider_state") not in STOPPED or money(fresh.get("current_total_usd_per_hour")) != 0:
                raise SafetyFailure("Exact pod is no longer stopped at zero hourly charge")
            if not 0 <= (self.now() - time_value(fresh.get("observed_at"))).total_seconds() <= 5:
                raise SafetyFailure("Current independent observation is stale")
            preflight = self.inner_control.verify_stopped_preflight(grant, current=now)
            if (preflight.get("pod_id") != self.pod_id or preflight.get("verified") is not True
                    or preflight.get("authorization_sha256") != sha(inner_authorization)):
                raise SafetyFailure("Exact-pod frozen control preflight was not verified")
            self.verify_guard()
            self.before_start()
            # Preflight/read/guard verification may take time. Re-establish freshness
            # at the actual dispatch boundary, not against the earlier saved clock.
            dispatch_now = self.now()
            current_grant = self.inner.validate_user_approval(inner_authorization, EXECUTION_SHA, current=dispatch_now)
            current_observed = private_json(stopped_observation)
            if current_grant != grant or current_observed != observed:
                raise SafetyFailure("Inner authorization or stopped observation changed during gates")
            self.inner.verify_provider_observation(grant, observed, current=dispatch_now)
            if ((dispatch_now - start).total_seconds() > 5
                    or (dispatch_now - at).total_seconds() > 600):
                raise SafetyFailure("Resume-request timestamp or stopped observation expired during gates")
            self.claim_operation("resume")
            self.event("resume-attempt", {"pod_id": self.pod_id, "machine_id": self.machine_id,
                "inner_authorization_sha256": sha(inner_authorization),
                "stopped_observation_sha256": sha(stopped_observation),
                "outer_intent_sha256": self.intent_sha,
                "accrued_upper_usd": str(self.accrued_upper),
                "inner_maximum_usd": grant["maximum_usd"], "inference_unlocked": False})
            self._resume_attempted = True
            try:
                self.cost_bound_verified = False
                self.provider.resume(self.pod_id, timeout=self.timeout())
                after = self.provider.read(self.pod_id, timeout=self.timeout())
                self.resource(after)
                self.cost_bound_verified = True
                if after.get("provider_state") != "RUNNING":
                    raise SafetyFailure("Resume did not establish RUNNING state")
                self.event("resume-identity-verified", {"pod_id": self.pod_id,
                    "machine_id": self.machine_id, "inference_unlocked": False})
                if self.handoff is not None:
                    self.handoff.run(self, grant)
            finally:
                # Completion, failed/ambiguous resume, changed host and inner failure
                # all stop immediately; no retry/repair or deadline extension.
                self.stop_exact()
            if not self.shutdown_receipts_durable:
                raise SafetyFailure("Stopped pod verified but shutdown receipt durability failed")
            self.phase = "STOPPED_LOCAL_LIFECYCLE_COMPLETE"
        except Exception:
            self.phase = "FAILED"
            raise


def create_live_controller(*, output=None, authorization=None, key_path=None,
                           inner_authorization=None, stop_preflight=None,
                           cli_style="modern", cli_executable="runpodctl", handoff=None):
    """Production assembly. Disabled gate runs before any credential or network access.

    Retain this session object for the one later resume. Process death is terminal;
    the external worker stops the resource, never restores/retries the controller.
    """
    from live_policy import require_live
    require_live("provisioning_enabled")
    require_live("external_shutdown_enabled")
    from provider_transport import RunPodProvider, RunPodObserver
    from external_shutdown import ExternalGuard as WorkerGuard
    from handoff import FrozenInnerControl
    # Do not pay to discover a readback limitation already known from the API
    # contract. A verified observation integration is required before enablement.
    if not RunPodObserver.proves_actual_state_rates_and_resources:
        raise SafetyFailure("Provider observation cannot prove actual state, split rates and VRAM")
    holder = {}
    def deadline():
        return time_value(holder["session"].intent["outer_provider_deadline"])
    provider = RunPodProvider(key_path, deadline, mutation_enabled=True)
    observer = RunPodObserver(key_path, deadline)
    guard = WorkerGuard(output, key_path)
    preflight = FrozenInnerControl(stop_preflight, inner_authorization, cli_style, cli_executable)
    session = LocalController(output, authorization, provider, observer, guard, preflight,
                              lambda: datetime.now(timezone.utc), _allow_live=True, handoff=handoff)
    holder["session"] = session
    return session


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Requires separately reviewed live enablement")
    args = parser.parse_args(argv)
    # There is no automatic approval writer or CLI shortcut around the factory.
    if args.execute:
        from live_policy import require_live
        try:
            require_live("provisioning_enabled")
        except PermissionError:
            pass
    print(json.dumps({"status": "LIVE_EXECUTION_UNAVAILABLE", "execution_enabled": False,
        "provider_calls": 0, "blocked": ["all committed live switches are false",
            "unverified provider actual-state/billing fields",
            "co-located inner watchdog, cache and bootstrap prerequisites"]}))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
