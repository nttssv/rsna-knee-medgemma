from datetime import datetime, timedelta, timezone
import http.server
import importlib.util
import json
from pathlib import Path
import sys
import threading

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/provider_transport.py"
spec = importlib.util.spec_from_file_location("provider_transport_test", PATH)
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
NAME = "qwen-provision-" + "a" * 32


def deadline():
    return datetime.now(timezone.utc) + timedelta(seconds=30)


def raw_pod(**updates):
    return {"id": "synthetic-pod", "name": NAME, "machineId": "synthetic-machine",
        "machine": {"secureCloud": True, "dataCenterId": "CA-MTL-1", "gpuTypeId": "NVIDIA A40"},
        "gpu": {"count": 1}, "containerDiskInGb": 80, "volumeInGb": 0,
        "networkVolume": None, "image": transport.IMAGE, "dockerStartCmd": transport.STARTUP,
        "templateId": None, "desiredStatus": "EXITED", "costPerHr": .49, **updates}


def test_exact_request_has_no_alternate_or_model_start():
    body = transport.create_body({**transport.FIXED, "name": NAME})
    assert body["gpuTypeIds"] == ["NVIDIA A40"]
    assert body["dataCenterIds"] == ["CA-MTL-1"]
    assert body["dockerStartCmd"] == ["/bin/sleep", "infinity"]
    assert body["volumeInGb"] == 0 and body["networkVolumeId"] is None
    assert "dockerEntrypoint" not in body  # Image ENTRYPOINT remains; documented blocker.


@pytest.mark.parametrize("field,value", [("gpu_name", "NVIDIA L40"), ("region", "US-WA-1"),
    ("gpu_count", True), ("gpu_count", 2), ("cloud_type", "COMMUNITY"),
    ("container_disk_gb", 100), ("persistent_volume_gb", 1), ("network_volume_id", "volume"),
    ("image_ref", "mutable:latest"), ("startup_command", ["python", "train.py"]),
    ("template_id", "template"), ("name", "short")])
def test_request_drift_rejected(field, value):
    with pytest.raises(transport.TransportFailure):
        transport.create_body({**transport.FIXED, "name": NAME, field: value})


@pytest.mark.parametrize("field", ["network_volume_id", "gpu_name", "template_id"])
def test_missing_request_field_rejected(field):
    request = {**transport.FIXED, "name": NAME}
    del request[field]
    with pytest.raises(transport.TransportFailure):
        transport.create_body(request)


def test_provider_desired_state_and_price_never_invent_zero_billing():
    observed = transport.normalize_pod(raw_pod(current_total_usd_per_hour=0,
        provider_state="STOPPED", gpu_vram_gb=48), observed_at=deadline().isoformat())
    assert observed["provider_desired_state"] == "EXITED"
    assert observed["provider_state"] == "UNVERIFIED"
    assert observed["current_total_usd_per_hour"] is None
    assert observed["actual_compute_usd_per_hour"] is None
    assert observed["actual_storage_usd_per_hour"] is None
    assert observed["gpu_vram_gb"] is None
    assert len(observed["observation_blockers"]) == 4


def test_missing_network_volume_is_not_inferred_unattached():
    raw = raw_pod()
    del raw["networkVolume"]
    assert transport.normalize_pod(raw, observed_at=deadline().isoformat())["network_volume_id"] == "UNVERIFIED"


@pytest.mark.parametrize("operation", ["create", "resume", "stop"])
@pytest.mark.parametrize("instance_enabled", [False, True])
def test_all_mutations_disabled_without_reading_key_or_http(monkeypatch, operation, instance_enabled):
    provider = transport.RunPodProvider(Path("/nonexistent-private-key"), deadline(), mutation_enabled=instance_enabled)
    monkeypatch.setattr(transport.network, "supervise", lambda *a, **k: pytest.fail("network accessed"))
    with pytest.raises(transport.TransportFailure, match="disabled"):
        getattr(provider, operation)({**transport.FIXED, "name": NAME} if operation == "create" else "synthetic-pod", timeout=10)


def test_single_create_lost_reply_cannot_retry(monkeypatch):
    monkeypatch.setattr(transport, "LIVE_MUTATIONS_ENABLED", True)
    provider = transport.RunPodProvider("unused", deadline(), mutation_enabled=True)
    calls = []
    def fail(*args, **kwargs):
        calls.append(args)
        raise transport.TransportFailure("synthetic lost reply")
    monkeypatch.setattr(provider, "_request", fail)
    with pytest.raises(transport.TransportFailure):
        provider.create({**transport.FIXED, "name": NAME}, timeout=10)
    with pytest.raises(transport.TransportFailure, match="second create"):
        provider.create({**transport.FIXED, "name": NAME}, timeout=10)
    assert len(calls) == 1


def test_single_resume_lost_reply_cannot_retry(monkeypatch):
    monkeypatch.setattr(transport, "LIVE_MUTATIONS_ENABLED", True)
    provider = transport.RunPodProvider("unused", deadline(), mutation_enabled=True)
    calls = []
    def fail(*args, **kwargs):
        calls.append(args)
        raise transport.TransportFailure("synthetic lost reply")
    monkeypatch.setattr(provider, "_request", fail)
    with pytest.raises(transport.TransportFailure):
        provider.resume("synthetic-pod", timeout=10)
    with pytest.raises(transport.TransportFailure, match="second resume"):
        provider.resume("synthetic-pod", timeout=10)
    assert len(calls) == 1


def test_exact_name_reconciliation_preserves_ambiguous_matches(monkeypatch):
    provider = transport.RunPodProvider("unused", deadline())
    monkeypatch.setattr(provider, "_request", lambda *a, **k: {"body": [raw_pod(), raw_pod(id="other")],
        "received_at": deadline().isoformat()})
    assert len(provider.find_by_name(NAME, timeout=5)) == 2


def test_observer_uses_fresh_separate_get(monkeypatch):
    seen = []
    def fake(command, payload, **kwargs):
        seen.append(payload)
        return {"ok": True, "body": raw_pod(), "received_at": deadline().isoformat()}
    monkeypatch.setattr(transport.network, "supervise", fake)
    provider = transport.RunPodProvider("unused", deadline())
    observer = transport.RunPodObserver("unused", deadline())
    provider.read("synthetic-pod", timeout=5)
    observer.read("synthetic-pod", timeout=5)
    assert len(seen) == 2 and all(x["method"] == "GET" for x in seen)
    assert not hasattr(observer, "stop")


def test_deadline_never_resets():
    value = [deadline()]
    provider = transport.RunPodProvider("unused", lambda: value[0])
    provider._deadline()
    value[0] += timedelta(seconds=1)
    with pytest.raises(transport.TransportFailure, match="cannot change"):
        provider._deadline()


@pytest.fixture
def server():
    calls = []
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(self.path)
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/must-not-follow")
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            if self.path == "/oversized":
                self.wfile.write(b"x" * (transport.MAX_RESPONSE + 1))
            elif self.path == "/echo":
                self.wfile.write(json.dumps({"credential": "synthetic-test-credential"}).encode())
            else:
                self.wfile.write(json.dumps({"value": 7}).encode())
        def log_message(self, *args):
            pass
    instance = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield instance, calls
    instance.shutdown()
    instance.server_close()
    thread.join(timeout=1)


def call_local(server, tmp_path, path):
    key = tmp_path / "fake-key"
    key.write_text("synthetic-test-credential")
    key.chmod(0o600)
    return transport.network.supervise([sys.executable, str(PATH), "--http-worker"],
        {"method": "GET", "url": f"http://127.0.0.1:{server[0].server_port}{path}",
         "key_path": str(key), "socket_timeout": 2, "loopback_test": True},
        timeout=3, deadline=deadline())


def test_real_supervised_localhost_get(server, tmp_path):
    assert call_local(server, tmp_path, "/ok")["body"] == {"value": 7}
    assert server[1] == ["/ok"]


@pytest.mark.parametrize("path", ["/redirect", "/oversized", "/echo"])
def test_redirect_size_and_secret_echo_fail_without_retry(server, tmp_path, path):
    result = call_local(server, tmp_path, path)
    assert result == {"ok": False, "error": "provider_request_failed"}
    assert server[1] == [path]
    assert "credential" not in json.dumps(result)


def test_private_key_requires_regular_owner_0600_and_no_symlink(tmp_path):
    key = tmp_path / "fake-key"
    key.write_text("synthetic")
    key.chmod(0o644)
    with pytest.raises(transport.TransportFailure):
        transport._private_key(key)
    key.chmod(0o600)
    link = tmp_path / "link"
    link.symlink_to(key)
    with pytest.raises(OSError):
        transport._private_key(link)
    provider = transport.RunPodProvider(link, deadline())
    with pytest.raises(OSError):
        provider.assert_credentials_private()
