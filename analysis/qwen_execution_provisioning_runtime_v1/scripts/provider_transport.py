"""One-request RunPod REST transport, disabled for every live mutation by default.

Official contract: https://docs.runpod.io/api-reference/pods/POST/pods and
GET/pods/podId, GET/pods, POST/pods/podId/start, POST/pods/podId/stop.
The documented REST representation does not prove actual stopped state, current
zero billing, separate storage rate, or GPU VRAM. Those are explicit blockers,
never manufactured from the creation request, desiredStatus, or a price list.
"""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import urllib.error
import urllib.parse
import urllib.request

_spec = importlib.util.spec_from_file_location("qwen_network_supervisor", Path(__file__).with_name("network_supervisor.py"))
network = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(network)

LIVE_MUTATIONS_ENABLED = False
API_BASE = "https://rest.runpod.io/v1"
IMAGE = "runpod/pytorch@sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851"
STARTUP = ["/bin/sleep", "infinity"]
FIXED = {"cloud_type": "SECURE", "region": "CA-MTL-1", "gpu_name": "NVIDIA A40",
    "gpu_count": 1, "gpu_vram_gb": 48, "container_disk_gb": 80,
    "persistent_volume_gb": 0, "network_volume_id": None, "image_ref": IMAGE,
    "startup_command": STARTUP, "template_id": None}
POD_ID = re.compile(r"[A-Za-z0-9_-]{1,100}\Z")
INTENT_NAME = re.compile(r"qwen-provision-[0-9a-f]{32}\Z")
MAX_RESPONSE = 1024 * 1024
READ_QUERY = "?includeMachine=true&includeNetworkVolume=true&includeSavingsPlans=true"


class TransportFailure(RuntimeError):
    pass


def _json_object_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON key")
        value[key] = item
    return value


def _private_key(path: Path) -> str:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != os.getuid() or not 1 <= info.st_size <= 4096):
            raise TransportFailure("Private owner-only 0600 credential file required")
        key = os.read(fd, 4097).decode().strip()
        if not key or not re.fullmatch(r"[A-Za-z0-9_-]+", key):
            raise TransportFailure("Credential format invalid")
        return key
    finally:
        os.close(fd)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise TransportFailure("Redirect refused")


def _http_worker(payload: dict) -> dict:
    """Single HTTP operation; called only inside the supervised worker process."""
    method, url = payload.get("method"), payload.get("url")
    parsed = urllib.parse.urlsplit(url)
    local_test = (payload.get("loopback_test") is True and parsed.scheme == "http"
                  and parsed.hostname == "127.0.0.1" and not parsed.username
                  and not parsed.password)
    if not local_test and not (parsed.scheme == "https" and parsed.netloc == "rest.runpod.io"
                              and parsed.path.startswith("/v1/pods") and not parsed.fragment):
        raise TransportFailure("Provider endpoint rejected")
    if method not in ("GET", "POST"):
        raise TransportFailure("HTTP method rejected")
    if method == "POST" and (not LIVE_MUTATIONS_ENABLED or payload.get("mutation_enabled") is not True):
        raise TransportFailure("Live mutations remain disabled")
    if parsed.username or parsed.password or parsed.fragment:
        raise TransportFailure("Invalid endpoint")
    key = _private_key(Path(payload["key_path"]))
    data = None if method == "GET" else json.dumps(payload.get("body", {}), allow_nan=False).encode()
    request = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": "Bearer " + key, "Accept": "application/json", "Content-Type": "application/json",
        "Cache-Control": "no-cache, no-store", "User-Agent": "rsna-qwen-provisioning/1.0"})
    # Disable proxy environment and automatic redirects. There is no retry loop.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(request, timeout=payload["socket_timeout"]) as response:
            if response.status not in (200, 201, 204):
                raise TransportFailure("Provider HTTP status rejected")
            if response.headers.get("Content-Encoding", "identity") != "identity":
                raise TransportFailure("Encoded response rejected")
            raw = response.read(MAX_RESPONSE + 1)
            if len(raw) > MAX_RESPONSE:
                raise TransportFailure("Provider response exceeds bound")
            # A response that echoes the credential must never reach a receipt.
            if key.encode() in raw:
                raise TransportFailure("Provider response rejected")
            value = None if not raw else json.loads(raw, object_pairs_hook=_json_object_pairs,
                parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
            return {"ok": True, "status": response.status, "body": value,
                    "received_at": datetime.now(timezone.utc).isoformat()}
    except Exception:
        # No URL, raw response, headers, error body, or original exception crosses
        # the pipe. A timed-out mutation has an unknown outcome and is not retried.
        raise TransportFailure("Provider request failed or response unverified") from None


def normalize_pod(raw: dict, *, observed_at: str) -> dict:
    """Map documented fields without inventing unavailable evidence.

    This function intentionally returns partial observations so a uniquely named
    pod can still be identified for emergency cleanup when resource fields drift.
    The controller rejects missing fields/rates before any resume or inference.
    """
    if not isinstance(raw, dict):
        raise TransportFailure("Provider pod response must be an object")
    machine = raw.get("machine") if isinstance(raw.get("machine"), dict) else {}
    gpu = raw.get("gpu") if isinstance(raw.get("gpu"), dict) else {}
    secure = machine.get("secureCloud")
    network_volume = raw.get("networkVolume", "MISSING")
    network_id = None if network_volume is None else (
        network_volume.get("id") if isinstance(network_volume, dict) else "UNVERIFIED")
    result = {
        "pod_id": raw.get("id"), "name": raw.get("name"), "machine_id": raw.get("machineId"),
        "observed_at": observed_at,
        "cloud_type": "SECURE" if secure is True else "COMMUNITY" if secure is False else None,
        "region": machine.get("dataCenterId"), "gpu_name": machine.get("gpuTypeId"),
        "gpu_count": gpu.get("count"), "gpu_vram_gb": None,
        "container_disk_gb": raw.get("containerDiskInGb"), "persistent_volume_gb": raw.get("volumeInGb"),
        "network_volume_id": network_id, "image_ref": raw.get("image"),
        "startup_command": raw.get("dockerStartCmd"), "template_id": raw.get("templateId", "MISSING"),
        "provider_desired_state": raw.get("desiredStatus"), "provider_state": "UNVERIFIED",
        "reported_running_cost_per_hour": raw.get("costPerHr"),
        "reported_adjusted_running_cost_per_hour": raw.get("adjustedCostPerHr"),
        "actual_compute_usd_per_hour": None, "actual_storage_usd_per_hour": None,
        "current_total_usd_per_hour": None,
        "observation_blockers": ["actual_provider_state_not_exposed", "current_total_billing_not_exposed",
            "separate_compute_storage_rates_not_exposed", "gpu_vram_not_exposed"],
    }
    # A supplied unknown JSON field named like our output does not become trusted.
    return result


def create_body(request: dict) -> dict:
    if set(request) != set(FIXED) | {"name"}:
        raise TransportFailure("Create request fields differ from reviewed allocation")
    for key, expected in FIXED.items():
        if request[key] != expected or type(request[key]) is not type(expected):
            raise TransportFailure("Create allocation differs from reviewed resource")
    if not isinstance(request["name"], str) or not INTENT_NAME.fullmatch(request["name"]):
        raise TransportFailure("Exact high-entropy intent name required")
    return {"name": request["name"], "cloudType": "SECURE", "computeType": "GPU",
        "gpuTypeIds": ["NVIDIA A40"], "gpuCount": 1, "dataCenterIds": ["CA-MTL-1"],
        "allowedCudaVersions": ["12.8"], "containerDiskInGb": 80, "volumeInGb": 0,
        "networkVolumeId": None, "imageName": IMAGE, "templateId": None,
        "dockerStartCmd": STARTUP.copy(), "env": {},
        "ports": [], "interruptible": False, "locked": False, "globalNetworking": False,
        "supportPublicIp": False}


class RunPodProvider:
    synthetic_only = False

    def __init__(self, key_path: Path, deadline=None, *, mutation_enabled=False):
        # Lexical absolute path only: resolve() would follow a credential symlink
        # before the worker's O_NOFOLLOW check could reject it.
        self.key_path = Path(os.path.abspath(key_path))
        self._deadline_source = deadline
        self._bound_deadline = None
        self.mutation_enabled = mutation_enabled
        self._create_attempted = False
        self._resume_attempted = False

    def assert_credentials_private(self):
        _private_key(self.key_path)  # Never return or retain the secret.

    def configure_deadline(self, deadline):
        if self._bound_deadline is not None and deadline != self._bound_deadline:
            raise TransportFailure("Outer transport deadline cannot change")
        if not isinstance(deadline, datetime) or deadline.tzinfo is None:
            raise TransportFailure("Aware outer deadline required")
        self._bound_deadline = deadline
        self._deadline_source = deadline

    def _deadline(self):
        value = self._deadline_source() if callable(self._deadline_source) else self._deadline_source
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise TransportFailure("Immutable outer deadline is not bound")
        if self._bound_deadline is None:
            self._bound_deadline = value
        if value != self._bound_deadline:
            raise TransportFailure("Outer transport deadline cannot change")
        return value

    def _mutation_gate(self):
        if not LIVE_MUTATIONS_ENABLED or self.mutation_enabled is not True:
            raise TransportFailure("Live mutations remain disabled")

    def _request(self, method, path, *, timeout, body=None):
        if method != "GET":
            self._mutation_gate()
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 30:
            raise TransportFailure("Network call requires a bounded timeout <=30 seconds")
        payload = {"method": method, "url": API_BASE + path, "key_path": str(self.key_path),
            "socket_timeout": float(timeout), "mutation_enabled": self.mutation_enabled, "body": body}
        try:
            envelope = network.supervise([sys.executable, str(Path(__file__).resolve()), "--http-worker"],
                payload, timeout=timeout, deadline=self._deadline())
        except Exception:
            raise TransportFailure("Provider call failed; outcome may be unknown") from None
        if envelope.get("ok") is not True:
            raise TransportFailure("Provider call failed; outcome may be unknown")
        return envelope

    def find_by_name(self, name, *, timeout):
        if not isinstance(name, str) or not INTENT_NAME.fullmatch(name):
            raise TransportFailure("Exact intent name required")
        envelope = self._request("GET", "/pods" + READ_QUERY, timeout=timeout)
        values = envelope.get("body")
        if not isinstance(values, list) or any(not isinstance(x, dict) for x in values):
            raise TransportFailure("Complete pod list was not returned")
        return [normalize_pod(x, observed_at=envelope["received_at"]) for x in values if x.get("name") == name]

    def read(self, pod_id, *, timeout):
        if not isinstance(pod_id, str) or not POD_ID.fullmatch(pod_id):
            raise TransportFailure("Exact pod ID required")
        envelope = self._request("GET", "/pods/" + pod_id + READ_QUERY, timeout=timeout)
        raw = envelope.get("body")
        if not isinstance(raw, dict) or raw.get("id") != pod_id:
            raise TransportFailure("Provider read returned a different pod")
        return normalize_pod(raw, observed_at=envelope["received_at"])

    def create(self, request, *, timeout):
        self._mutation_gate()
        if self._create_attempted:
            raise TransportFailure("A second create is forbidden")
        body = create_body(request)
        self._create_attempted = True  # Lost response remains consumed.
        envelope = self._request("POST", "/pods", timeout=timeout, body=body)
        value = envelope.get("body")
        pod = value.get("id") if isinstance(value, dict) else None
        if not isinstance(pod, str) or not POD_ID.fullmatch(pod):
            raise TransportFailure("Create response did not establish pod ID")
        return pod

    def stop(self, pod_id, *, timeout):
        if not isinstance(pod_id, str) or not POD_ID.fullmatch(pod_id):
            raise TransportFailure("Exact pod ID required")
        self._request("POST", "/pods/" + pod_id + "/stop", timeout=timeout, body={})

    def resume(self, pod_id, *, timeout):
        self._mutation_gate()
        if self._resume_attempted:
            raise TransportFailure("A second resume is forbidden")
        if not isinstance(pod_id, str) or not POD_ID.fullmatch(pod_id):
            raise TransportFailure("Exact pod ID required")
        self._resume_attempted = True
        self._request("POST", "/pods/" + pod_id + "/start", timeout=timeout, body={})


class RunPodObserver:
    """Separate uncached GET path; cannot create, resume, or stop resources."""
    synthetic_only = False
    proves_actual_state_rates_and_resources = False

    def __init__(self, key_path, deadline=None):
        self._reader = RunPodProvider(key_path, deadline)

    def configure_deadline(self, deadline):
        self._reader.configure_deadline(deadline)

    def read(self, pod_id, *, timeout):
        return self._reader.read(pod_id, timeout=timeout)


def main():
    if sys.argv[1:] != ["--http-worker"]:
        return 2
    try:
        raw = sys.stdin.buffer.read(network.MAX_INPUT_BYTES + 1)
        if len(raw) > network.MAX_INPUT_BYTES:
            return 2
        payload = json.loads(raw, object_pairs_hook=_json_object_pairs)
        result = _http_worker(payload)
    except Exception:
        result = {"ok": False, "error": "provider_request_failed"}
    sys.stdout.write(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
