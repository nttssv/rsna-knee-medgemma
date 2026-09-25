"""One-shot, exact-pod RunPod stop backstop for the approved running-pod session."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request

POD_RE = re.compile(r"[A-Za-z0-9_-]{1,100}\Z")
API = "https://api.runpod.io/graphql"
MAX_BUDGET = 3.0
MAX_COMPUTE = 0.84
MAX_STORAGE = 0.012
MAX_SECONDS = 10800
STOP_OFFSET = 10500
EXPECTED_POD_ID_SHA256 = hashlib.sha256(b"rqnenq3mpu0i2g").hexdigest()


class GuardError(RuntimeError):
    pass


def parse_time(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise GuardError("Timezone-aware timestamp required")
    return result.astimezone(timezone.utc)


def validate_intent(intent: dict, now: datetime | None = None) -> datetime:
    required = {"pod_id", "pod_name", "t0", "hard_deadline", "shutdown_at",
                "compute_usd_per_hour", "storage_usd_per_hour", "maximum_usd",
                "execution_plan_sha256", "source_sha256", "cancellation_path"}
    if not isinstance(intent, dict) or set(intent) != required:
        raise GuardError("External stop intent schema mismatch")
    if intent["pod_name"] != "low_brown_viper" or not POD_RE.fullmatch(intent["pod_id"]):
        raise GuardError("Exact selected pod identity is required")
    if hashlib.sha256(intent["pod_id"].encode()).hexdigest() != EXPECTED_POD_ID_SHA256:
        raise GuardError("Pod ID is not the currently approved low_brown_viper pod")
    cancel = intent["cancellation_path"]
    if not isinstance(cancel, str) or not Path(cancel).is_absolute():
        raise GuardError("Absolute private cancellation marker path required")
    start = parse_time(intent["t0"])
    hard = parse_time(intent["hard_deadline"])
    stop = parse_time(intent["shutdown_at"])
    if hard != start + timedelta(seconds=MAX_SECONDS) or stop != start + timedelta(seconds=STOP_OFFSET):
        raise GuardError("Immutable 3-hour deadline / T+175 shutdown binding mismatch")
    for key in ("execution_plan_sha256", "source_sha256"):
        if not isinstance(intent[key], str) or not re.fullmatch(r"[0-9a-f]{64}", intent[key]):
            raise GuardError("Execution provenance hash invalid")
    rates = (intent["compute_usd_per_hour"], intent["storage_usd_per_hour"])
    if any(type(v) not in (int, float) for v in rates) or not 0 < rates[0] <= MAX_COMPUTE or not 0 <= rates[1] <= MAX_STORAGE:
        raise GuardError("Provider rates exceed the approved ceiling")
    if type(intent["maximum_usd"]) not in (int, float) or not 0 < intent["maximum_usd"] <= MAX_BUDGET:
        raise GuardError("Session budget exceeds approval")
    if (rates[0] + rates[1]) * 3 > intent["maximum_usd"]:
        raise GuardError("Three-hour maximum cost exceeds the session budget")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None or current >= hard:
        raise GuardError("Hard provider deadline reached")
    return stop


def read_private_json(path: Path) -> dict:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.getuid():
        raise GuardError("Private owner-only 0600 file required")
    return json.loads(path.read_text())


def read_private_key(path: Path) -> str:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.getuid():
        raise GuardError("Private owner-only 0600 API key file required")
    key = path.read_text().strip()
    if not key or not re.fullmatch(r"[A-Za-z0-9_-]{20,256}", key):
        raise GuardError("API key format invalid")
    return key


def stop_once(pod_id: str, key_path: Path, *, opener=None) -> dict:
    """Issue exactly one bounded provider stop request; never retry an unknown result."""
    if not POD_RE.fullmatch(pod_id) or hashlib.sha256(pod_id.encode()).hexdigest() != EXPECTED_POD_ID_SHA256:
        raise GuardError("Pod ID invalid")
    key = read_private_key(key_path)
    query = ('mutation { podStop(input: { podId: "' + pod_id + '" }) '
             '{ id desiredStatus } }')
    request = urllib.request.Request(API, data=json.dumps({"query": query}).encode(), method="POST", headers={
        "Authorization": "Bearer " + key, "Accept": "application/json",
        "Content-Type": "application/json", "User-Agent": "rsna-qwen-running-session/1.0"})
    if opener is None:
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                raise GuardError("Provider redirect refused")
        open_fn = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect()).open
    else:
        open_fn = opener
    try:
        with open_fn(request, timeout=20) as response:
            if response.status != 200:
                raise GuardError("Provider stop response was not successful")
            body = response.read(4096)
            if key.encode() in body:
                raise GuardError("Provider response unexpectedly echoed a credential")
            payload = json.loads(body)
            stopped = payload.get("data", {}).get("podStop", {})
            if stopped.get("id") != pod_id or stopped.get("desiredStatus") not in ("EXITED", "STOPPED"):
                raise GuardError("Provider GraphQL response did not bind the exact pod stop")
            return {"status": "stop_request_accepted", "pod_id": pod_id,
                    "provider_desired_status": stopped["desiredStatus"],
                    "received_at": datetime.now(timezone.utc).isoformat()}
    except GuardError:
        raise
    except Exception as exc:
        # A timeout may mean the stop succeeded; never send a second mutation.
        raise GuardError("Single provider stop request failed or has unknown outcome") from None


def worker(intent_path: Path, key_path: Path, receipt_path: Path) -> int:
    intent = read_private_json(intent_path)
    stop_at = validate_intent(intent)
    result = {"intent_sha256": hashlib.sha256(intent_path.read_bytes()).hexdigest(),
              "pod_id": intent["pod_id"], "requested_at": datetime.now(timezone.utc).isoformat()}
    cancel_path = Path(intent["cancellation_path"])
    while datetime.now(timezone.utc) < stop_at:
        if cancel_path.exists():
            result.update(status="cancelled_after_operator_verified_stop",
                cancelled_at=datetime.now(timezone.utc).isoformat())
            receipt_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as stream:
                json.dump(result, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            return 0
        time.sleep(min(10, max(0, (stop_at - datetime.now(timezone.utc)).total_seconds())))
    if cancel_path.exists():
        result.update(status="cancelled_after_operator_verified_stop",
            cancelled_at=datetime.now(timezone.utc).isoformat())
        receipt_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return 0
    try:
        result.update(stop_once(intent["pod_id"], key_path))
        result["status"] = "stop_request_accepted"
    except Exception as exc:
        result.update(status="stop_request_failed_or_unknown", error_type=type(exc).__name__)
    receipt_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return 0 if result["status"] == "stop_request_accepted" else 1


def arm(intent_path: Path, key_path: Path, receipt_path: Path) -> int:
    intent = read_private_json(intent_path)
    validate_intent(intent)
    read_private_key(key_path)
    receipt_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    log_path = receipt_path.with_suffix(".log")
    with log_path.open("xb") as log:
        os.chmod(log_path, 0o600)
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker",
            str(intent_path), str(key_path), str(receipt_path)], stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    return process.pid


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--worker":
        raise SystemExit(worker(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])))
    if len(sys.argv) == 5 and sys.argv[1] == "--arm":
        raise SystemExit(arm(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])))
    raise SystemExit("Use the reviewed operator entry point to arm the stop guard")
