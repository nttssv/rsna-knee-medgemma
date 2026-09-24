"""Detached, intent-bound shutdown backstop. Live deployment is disabled.

The core is testable without network. A real worker uses the separately supervised
transport; it never creates or resumes a pod. OS identity plus a fresh private
socket challenge verifies liveness, rather than trusting a PID receipt.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import importlib.util
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import signal
import socket
import stat
import subprocess
import sys
import time

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import controller as c

LIVE_SHUTDOWN_ENABLED = False
STOP_ATTEMPTS = 3
MAX_CALL_SECONDS = 30.0
POLL_SECONDS = 1.0
ENV_SHA = "QWEN_OUTER_GUARD_CONFIG_SHA"
ENV_NONCE = "QWEN_OUTER_GUARD_NONCE"


def source_bundle():
    value = {path.name: c.sha(path) for path in sorted(HERE.parent.glob("*.py"))}
    value.update({"../configs/" + path.name: c.sha(path)
                  for path in sorted((HERE.parent.parent / "configs").glob("*.json"))})
    return value


def utcnow():
    return datetime.now(timezone.utc)


def private_directory(path):
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700 or info.st_uid != os.getuid():
        raise c.SafetyFailure("Private owner-only 0700 directory required")
    return path


def process_info(pid):
    """Read identity from the OS; never from the worker's own receipt.

    Linux is the production platform. The macOS branch supports local synthetic
    detached-process tests and checks the same command and nonce environment.
    """
    if type(pid) is not int or pid <= 0:
        raise c.SafetyFailure("Invalid PID")
    try:
        if sys.platform.startswith("linux"):
            root = Path(f"/proc/{pid}")
            fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] == "Z":
                raise c.SafetyFailure("Worker is a zombie")
            args = (root / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
            env = dict(value.split("=", 1) for value in (root / "environ").read_bytes().decode().split("\0") if "=" in value)
            return {"start": fields[19], "pgid": int(fields[2]), "sid": int(fields[3]),
                    "argv": args, "env": env, "uid": root.stat().st_uid}
        if sys.platform == "darwin":
            def ps(field, environment=False):
                command = ["/bin/ps", "-ww", "-p", str(pid), "-o", field + "="]
                if environment:
                    command.insert(1, "eww")
                result = subprocess.run(command, capture_output=True, text=True, timeout=2, check=True)
                return result.stdout.strip()
            command = ps("command")
            environment = ps("command", True)
            env = {}
            for key in (ENV_SHA, ENV_NONCE):
                matches = re.findall(r"(?:^|\s)" + key + r"=([^\s]+)(?:\s|$)", environment)
                if len(matches) == 1:
                    env[key] = matches[0]
            if not command or ps("stat").startswith("Z"):
                raise c.SafetyFailure("Worker is absent or a zombie")
            return {"start": ps("lstart"), "pgid": os.getpgid(pid), "sid": os.getsid(pid),
                    "argv": shlex.split(command), "env": env, "uid": int(ps("uid"))}
        raise c.SafetyFailure("OS process identity verification is unavailable")
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise c.SafetyFailure("OS process identity verification failed") from exc


def same_process(pid, start):
    try:
        return process_info(pid)["start"] == start
    except c.SafetyFailure:
        return False


def validate_intent(intent, intent_sha, intent_path):
    if c.sha(intent_path) != intent_sha or c.private_json(intent_path) != intent:
        raise c.SafetyFailure("Immutable outer intent differs")
    first = c.time_value(intent.get("first_create_requested_at"))
    deadline = c.time_value(intent.get("outer_provider_deadline"))
    stop = c.time_value(intent.get("outer_watchdog_stop_at"))
    if not 300 < (deadline - first).total_seconds() <= 3600 or stop != deadline - timedelta(seconds=300):
        raise c.SafetyFailure("Outer deadline/reserve binding differs")
    maximum = c.money(intent.get("maximum_usd"))
    if not 0 < maximum <= c.MAXIMUM or c.CEILING_PER_HOUR * (c.money((deadline - first).total_seconds()) / 3600) > maximum:
        raise c.SafetyFailure("Outer cumulative budget differs")
    if not re.fullmatch(r"qwen-provision-[0-9a-f]{32}", intent.get("intent_name", "")):
        raise c.SafetyFailure("Exact high-entropy intent name required")
    allocation = intent.get("allocation", {})
    for key, expected in c.FIXED.items():
        if allocation.get(key) != expected or type(allocation.get(key)) is not type(expected):
            raise c.SafetyFailure("Exact outer allocation differs")
    if not 0 < c.money(allocation.get("actual_compute_usd_per_hour")) <= c.money(.49) or c.money(allocation.get("actual_storage_usd_per_hour")) > c.money(.012):
        raise c.SafetyFailure("Outer selected rate ceiling differs")
    if intent.get("provisioning_proposal_sha256") != c.PROVISIONING_SHA or intent.get("controller_source_sha256") != c.sha(Path(c.__file__)):
        raise c.SafetyFailure("Reviewed source/proposal binding differs")


class ShutdownCore:
    """Only cleanup: no create/resume interface is accessed or required.

    Reconciliation reads may repeat when an accepted create is not yet visible.
    A unique owned ID is bound once; conflicting IDs fail closed. Stop requests
    total at most three across all ticks, with a fresh independent read afterward.
    Logging failure must never suppress cleanup of an already identified pod.
    """
    def __init__(self, output, intent, intent_sha, provider, observer, *, clock=utcnow,
                 allocator_alive=lambda: True, record=None):
        if provider is observer:
            raise c.SafetyFailure("Independent stop observation required")
        self.output = private_directory(output)
        validate_intent(intent, intent_sha, self.output / "intent.json")
        self.intent, self.intent_sha = intent, intent_sha
        self.provider, self.observer, self.clock = provider, observer, clock
        self.allocator_alive = allocator_alive
        self.record = record or self._record
        self.pod_id = None
        self.stop_attempts = 0
        self.sequence = 0
        self.ownership_cached = False
        self.emergency = False
        self.done = False
        self.result = None
        self.log_durable = True
        self.deadline = c.time_value(intent["outer_provider_deadline"])
        self.stop_at = c.time_value(intent["outer_watchdog_stop_at"])

    def _record(self, kind, fields):
        self.sequence += 1
        return c.write_exclusive(self.output / f"guard-{self.sequence:04d}-{kind}.json",
            {"kind": kind, "recorded_at": self.clock().isoformat(), "intent_sha256": self.intent_sha, **fields})

    def log(self, kind, fields=None):
        try:
            self.record(kind, fields or {})
        except Exception:
            self.log_durable = False

    def timeout(self):
        remaining = (self.deadline - self.clock()).total_seconds()
        if remaining <= 0:
            raise c.SafetyFailure("Hard provider deadline reached")
        return min(MAX_CALL_SECONDS, remaining)

    def ownership_evidence(self):
        """Only an empty preread followed by one fsynced create-attempt owns a name."""
        if self.ownership_cached:
            return True
        validate_intent(self.intent, self.intent_sha, self.output / "intent.json")
        prereads = list(self.output.glob("[0-9]*-intent-name-preread.json"))
        attempts = list(self.output.glob("[0-9]*-create-attempt.json"))
        if len(prereads) != 1 or len(attempts) != 1:
            return False
        pre, attempt = c.private_json(prereads[0]), c.private_json(attempts[0])
        if (pre.get("kind") != "intent-name-preread" or type(pre.get("matches")) is not int
                or pre["matches"] != 0 or attempt.get("kind") != "create-attempt"
                or attempt.get("intent_sha256") != self.intent_sha
                or type(attempt.get("maximum_attempts")) is not int or attempt["maximum_attempts"] != 1):
            raise c.SafetyFailure("Durable creation ownership evidence differs")
        first = c.time_value(self.intent["first_create_requested_at"])
        if not first <= c.time_value(pre["recorded_at"]) <= c.time_value(attempt["recorded_at"]) < self.stop_at:
            raise c.SafetyFailure("Creation ownership timeline differs")
        self.ownership_cached = True
        return True

    def fresh_identity(self, observed, pod_id=None):
        identity = observed.get("pod_id")
        if not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", identity):
            raise c.SafetyFailure("Invalid cleanup pod identity")
        if pod_id is not None and identity != pod_id:
            raise c.SafetyFailure("Cleanup pod identity drift")
        if observed.get("name") != self.intent["intent_name"]:
            raise c.SafetyFailure("Cleanup exact intent name differs")
        if not 0 <= (self.clock() - c.time_value(observed.get("observed_at"))).total_seconds() <= 5:
            raise c.SafetyFailure("Cleanup observation freshness differs")
        return identity

    def reconcile(self):
        candidates = self.provider.find_by_name(self.intent["intent_name"], timeout=self.timeout())
        if not isinstance(candidates, list) or len(candidates) != 1:
            self.log("ownership-unresolved", {"matches": len(candidates) if isinstance(candidates, list) else None})
            return False
        pod = self.fresh_identity(candidates[0], self.pod_id)
        self.pod_id = pod
        # Resource mismatch never authorizes use; exact name ownership is sufficient
        # to clean up the potentially billable allocation made by the sole create.
        self.log("cleanup-target", {"pod_id": pod})
        return True

    def stopped(self, observed):
        self.fresh_identity(observed, self.pod_id)
        return (observed.get("provider_state") in c.STOPPED
                and c.money(observed.get("current_total_usd_per_hour")) == 0)

    def tick(self):
        if self.done:
            return self.result
        now = self.clock()
        allocator_dead = self.emergency or not self.allocator_alive()
        if not self.ownership_cached:
            try:
                self.ownership_evidence()
            except Exception:
                if allocator_dead:
                    self.log("ownership-evidence-unreadable")
        if now < self.stop_at and not allocator_dead:
            return "ARMED"
        if now >= self.deadline:
            self.done, self.result = True, "SHUTDOWN_UNVERIFIED_DEADLINE"
            self.log("terminal", {"status": self.result, "pod_id": self.pod_id})
            return self.result
        try:
            if not self.ownership_evidence():
                # If the allocator died before its durable attempt, no create was
                # authorized to dispatch. A living allocator remains monitored.
                if allocator_dead:
                    self.done, self.result = True, "NO_CREATE_ATTEMPT"
                    self.log("terminal", {"status": self.result})
                    return self.result
                return "NO_CREATE_ATTEMPT"
            if not self.reconcile():
                return "OWNERSHIP_UNRESOLVED"
            if self.stop_attempts < STOP_ATTEMPTS:
                self.stop_attempts += 1
                self.log("stop-attempt", {"pod_id": self.pod_id, "attempt": self.stop_attempts})
                try:
                    self.provider.stop(self.pod_id, timeout=self.timeout())
                except Exception as exc:
                    self.log("stop-response-unconfirmed", {"error_type": type(exc).__name__})
            observed = self.observer.read(self.pod_id, timeout=self.timeout())
            if self.stopped(observed):
                self.done, self.result = True, "INDEPENDENTLY_STOPPED_ZERO_RATE"
                self.log("terminal", {"status": self.result, "pod_id": self.pod_id,
                    "provider_observation": observed, "receipts_durable_before_final_write": self.log_durable})
                return self.result
            self.log("stop-unverified", {"pod_id": self.pod_id, "attempts": self.stop_attempts})
            return "SHUTDOWN_UNVERIFIED"
        except Exception as exc:
            self.log("cleanup-error", {"error_type": type(exc).__name__, "pod_id": self.pod_id})
            return "SHUTDOWN_UNVERIFIED"


def _proof(challenge, config, pid, ownership_cached=False):
    message = json.dumps({"challenge": challenge, "pid": pid,
        "intent_sha256": config["intent_sha256"], "source_sha256": config["source_sha256"],
        "ownership_cached": ownership_cached}, sort_keys=True).encode()
    return hmac.new(bytes.fromhex(config["challenge_key"]), message, hashlib.sha256).hexdigest()


class ExternalGuard:
    """Start once, detached; verify before each controller dispatch.

    Production startup additionally requires the reviewed module live switch. No
    source/API flag can enable it in this implementation milestone. synthetic_state
    selects a local-only file adapter used solely by subprocess tests.
    """
    synthetic_only = False

    def __init__(self, output, key_path=None, *, synthetic_state=None):
        self.output = Path(output)
        self.key_path = Path(key_path) if key_path is not None else None
        self.synthetic_state = Path(synthetic_state) if synthetic_state is not None else None
        self.synthetic_only = self.synthetic_state is not None
        self.process = None
        self.config = None
        self.config_path = None
        self.info = None
        self.argv = None
        self.socket_path = None
        self.macos_interpreter = None

    def start(self, intent, intent_sha):
        if self.process is not None:
            raise c.SafetyFailure("External worker cannot be restarted")
        if not self.synthetic_only and not LIVE_SHUTDOWN_ENABLED:
            raise c.SafetyFailure("Live external shutdown startup remains disabled")
        if not self.synthetic_only and not sys.platform.startswith("linux"):
            raise c.SafetyFailure("Live external worker requires Linux process attestation")
        private_directory(self.output)
        validate_intent(intent, intent_sha, self.output / "intent.json")
        state = self.output / "external-guard"
        state.mkdir(mode=0o700)
        private_directory(state)
        allocator = process_info(os.getpid())
        self.macos_interpreter = allocator["argv"][0] if sys.platform == "darwin" else None
        nonce = secrets.token_hex(32)
        # Short private path avoids AF_UNIX limits under long workspace directories.
        socket_directory = Path("/tmp") / ("qwen-guard-" + secrets.token_hex(16))
        socket_directory.mkdir(mode=0o700)
        self.socket_path = socket_directory / "control.sock"
        self.config_path = state / "worker.json"
        snapshot = state / "transport-source"
        snapshot.mkdir(mode=0o700)
        snapshot_hashes = {}
        for name in ("provider_transport.py", "network_supervisor.py"):
            original = HERE.parent / name
            payload = original.read_bytes()
            destination = snapshot / name
            fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(fd, "wb", closefd=False) as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(fd)
            finally:
                os.close(fd)
            snapshot_hashes[name] = hashlib.sha256(payload).hexdigest()
        self.config = {"intent": intent, "intent_sha256": intent_sha,
            "output": str(self.output.resolve()), "source_sha256": c.sha(HERE),
            "source_bundle": source_bundle(), "transport_source": str(snapshot.resolve()),
            "transport_source_hashes": snapshot_hashes,
            "allocator_pid": os.getpid(), "allocator_start": allocator["start"],
            "socket_path": str(self.socket_path), "challenge_key": secrets.token_hex(32),
            "environment_nonce": nonce, "key_path": os.path.abspath(self.key_path) if self.key_path else None,
            "key_sha256": c.sha(self.key_path) if self.key_path else None,
            "synthetic_state": str(self.synthetic_state.resolve()) if self.synthetic_state else None}
        if not self.synthetic_only:
            from provider_transport import RunPodProvider, LIVE_MUTATIONS_ENABLED
            if not LIVE_MUTATIONS_ENABLED:
                raise c.SafetyFailure("Provider stop transport remains disabled")
            transport = RunPodProvider(self.key_path, c.time_value(intent["outer_provider_deadline"]))
            transport.assert_credentials_private()
        if snapshot_hashes != {name: self.config["source_bundle"].get(name) for name in snapshot_hashes}:
            raise c.SafetyFailure("Transport changed during private source snapshot")
        directory_fd = os.open(snapshot, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        config_sha = c.write_exclusive(self.config_path, self.config)
        self.argv = [sys.executable, "-I", str(HERE), "--worker", str(self.config_path)]
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
               ENV_SHA: config_sha, ENV_NONCE: nonce}
        # No credentials in argv/environment/stdout; stdin is disconnected. A new
        # session is independent of the allocating parent's process group.
        self.process = subprocess.Popen(self.argv, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
            start_new_session=True, close_fds=True)
        self.info = process_info(self.process.pid)
        end = time.monotonic() + 4
        while not self.socket_path.exists():
            if self.process.poll() is not None or time.monotonic() >= end:
                if self.process.poll() is None:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=2)
                self.socket_path.unlink(missing_ok=True)
                socket_directory.rmdir()
                raise c.SafetyFailure("Detached external worker did not become ready")
            time.sleep(.02)
        return self.process.pid

    def verify(self, intent, intent_sha256, *, current, require_create_attempt=False):
        if self.process is None:
            self.start(intent, intent_sha256)
        if self.process.poll() is not None:
            raise c.SafetyFailure("External shutdown worker died; no restart permitted")
        if self.config["intent"] != intent or self.config["intent_sha256"] != intent_sha256:
            raise c.SafetyFailure("External worker intent differs")
        if (c.private_json(self.config_path) != self.config or c.sha(HERE) != self.config["source_sha256"]
                or self.config["source_bundle"] != source_bundle()):
            raise c.SafetyFailure("External worker configuration/source drift")
        if self.key_path is not None and c.sha(self.key_path) != self.config["key_sha256"]:
            raise c.SafetyFailure("External worker credential identity drift")
        validate_intent(intent, intent_sha256, self.output / "intent.json")
        pid = self.process.pid
        info = process_info(pid)
        expected_env = {ENV_SHA: c.sha(self.config_path), ENV_NONCE: self.config["environment_nonce"]}
        command_matches = info["argv"] == self.argv
        if sys.platform == "darwin" and self.synthetic_only:
            command_matches = (info["argv"][0] == self.macos_interpreter and info["argv"][1:] == self.argv[1:])
        if (pid == os.getpid() or info["uid"] != os.getuid() or info["start"] != self.info["start"]
                or info["pgid"] != pid or info["sid"] != pid or not command_matches
                or any(info["env"].get(key) != value for key, value in expected_env.items())):
            raise c.SafetyFailure("External worker OS command/environment/session identity differs")
        private_directory(self.socket_path.parent)
        challenge = secrets.token_hex(32)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(2)
            connection.connect(str(self.socket_path))
            connection.sendall((challenge + "\n").encode())
            response = json.loads(connection.recv(4096).decode())
        ownership_cached = response.get("ownership_cached")
        if (type(ownership_cached) is not bool
                or not hmac.compare_digest(response.get("proof", ""), _proof(challenge, self.config, pid, ownership_cached))):
            raise c.SafetyFailure("External worker fresh challenge failed")
        if require_create_attempt and ownership_cached is not True:
            raise c.SafetyFailure("External worker has not cached durable create ownership")
        if not same_process(pid, info["start"]):
            raise c.SafetyFailure("External worker died during verification")
        now = utcnow()
        if not 0 <= (now - current).total_seconds() <= 5 or now >= c.time_value(intent["outer_watchdog_stop_at"]):
            raise c.SafetyFailure("External guard verification is stale or reserve reached")
        return {"pid": pid, "intent_sha256": intent_sha256, "intent_name": intent["intent_name"],
            "outer_provider_deadline": intent["outer_provider_deadline"],
            "outer_watchdog_stop_at": intent["outer_watchdog_stop_at"],
            "independent_of_allocating_process": True, "alive": True,
            "command_and_environment_verified": True, "stop_capable": True,
            "verified_at": now.isoformat(), "worker_source_sha256": c.sha(HERE),
            "worker_config_sha256": c.sha(self.config_path), "synthetic_only": self.synthetic_only,
            "durable_create_attempt_verified": ownership_cached}


class _SyntheticProvider:
    """Subprocess-test fixture only: local JSON, never a provider endpoint."""
    def __init__(self, path):
        self.path = Path(path)

    def find_by_name(self, name, *, timeout):
        state = c.private_json(self.path)
        return [{**pod, "observed_at": utcnow().isoformat()} for pod in state["pods"] if pod["name"] == name]

    def read(self, pod_id, *, timeout):
        state = c.private_json(self.path)
        return next({**pod, "observed_at": utcnow().isoformat()} for pod in state["pods"] if pod["pod_id"] == pod_id)

    def stop(self, pod_id, *, timeout):
        state = c.private_json(self.path)
        for pod in state["pods"]:
            if pod["pod_id"] == pod_id:
                pod.update(provider_state="STOPPED", current_total_usd_per_hour=0)
        state["stop_calls"] = state.get("stop_calls", 0) + 1
        temporary = self.path.with_name(self.path.name + ".write-" + secrets.token_hex(8))
        c.write_exclusive(temporary, state)
        os.replace(temporary, self.path)


def worker(config_path):
    config_path = Path(config_path)
    config = c.private_json(config_path)
    if os.environ.get(ENV_SHA) != c.sha(config_path) or os.environ.get(ENV_NONCE) != config["environment_nonce"]:
        raise c.SafetyFailure("Worker environment binding differs")
    if config["source_sha256"] != c.sha(HERE) or config["source_bundle"] != source_bundle():
        raise c.SafetyFailure("Worker source binding differs")
    output = private_directory(config["output"])
    validate_intent(config["intent"], config["intent_sha256"], output / "intent.json")
    snapshot = private_directory(config["transport_source"])
    for name, expected in config["transport_source_hashes"].items():
        if name not in {"provider_transport.py", "network_supervisor.py"} or c.sha(snapshot / name) != expected:
            raise c.SafetyFailure("Private shutdown transport source differs")
    if config.get("synthetic_state"):
        provider = _SyntheticProvider(config["synthetic_state"])
        observer = _SyntheticProvider(config["synthetic_state"])
    else:
        if not LIVE_SHUTDOWN_ENABLED:
            raise c.SafetyFailure("Live external shutdown remains disabled")
        if c.sha(Path(config["key_path"])) != config["key_sha256"]:
            raise c.SafetyFailure("Worker credential binding differs")
        spec = importlib.util.spec_from_file_location("guard_pinned_transport", snapshot / "provider_transport.py")
        transport = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(transport)
        provider = transport.RunPodProvider(Path(config["key_path"]), c.time_value(config["intent"]["outer_provider_deadline"]), mutation_enabled=True)
        observer = transport.RunPodObserver(Path(config["key_path"]), c.time_value(config["intent"]["outer_provider_deadline"]))
    started_wall, started_monotonic = utcnow(), time.monotonic()
    def bounded_clock():
        return max(utcnow(), started_wall + timedelta(seconds=time.monotonic() - started_monotonic))
    core = ShutdownCore(output, config["intent"], config["intent_sha256"], provider, observer,
        clock=bounded_clock, allocator_alive=lambda: same_process(config["allocator_pid"], config["allocator_start"]))
    path = Path(config["socket_path"])
    private_directory(path.parent)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(path))
        path.chmod(0o600)
        server.listen(4)
        server.settimeout(POLL_SECONDS)
        try:
            while not core.done:
                try:
                    drifted = source_bundle() != config["source_bundle"] or c.private_json(config_path) != config
                except Exception:
                    drifted = True
                if drifted:
                    core.log("source-drift", {"emergency_cleanup": True})
                    core.emergency = True
                core.tick()
                if core.done:
                    break
                try:
                    connection, _ = server.accept()
                except socket.timeout:
                    continue
                with connection:
                    connection.settimeout(.25)
                    try:
                        challenge = connection.recv(256).decode().strip()
                        if re.fullmatch(r"[0-9a-f]{64}", challenge):
                            try:
                                core.ownership_evidence()
                            except Exception:
                                pass
                            response = {"ownership_cached": core.ownership_cached,
                                "proof": _proof(challenge, config, os.getpid(), core.ownership_cached)}
                            connection.sendall((json.dumps(response) + "\n").encode())
                    except (OSError, UnicodeError):
                        pass
        finally:
            path.unlink(missing_ok=True)
            try:
                path.parent.rmdir()
            except OSError:
                pass
    return 0 if core.result in {"INDEPENDENTLY_STOPPED_ZERO_RATE", "NO_CREATE_ATTEMPT"} else 2


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.execute or args.worker is None:
        print(json.dumps({"execution_enabled": False, "live_shutdown_enabled": False}))
        return 2
    try:
        return worker(args.worker)
    except Exception:
        # Never emit exception text that could contain credentials or provider payloads.
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
