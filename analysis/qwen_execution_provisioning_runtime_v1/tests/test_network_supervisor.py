from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/network_supervisor.py"
spec = importlib.util.spec_from_file_location("network_supervisor_test", PATH)
net = importlib.util.module_from_spec(spec)
spec.loader.exec_module(net)


def run(code, *, timeout=2, **kwargs):
    return net.supervise([sys.executable, "-c", code], {"value": 7}, timeout=timeout,
        deadline=datetime.now(timezone.utc) + timedelta(seconds=timeout), **kwargs)


def test_success_and_clean_environment(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "synthetic-secret")
    result = run('import json,sys,os; p=json.load(sys.stdin); print(json.dumps({"v":p["value"],"secret":os.environ.get("RUNPOD_API_KEY")}))')
    assert result == {"v": 7, "secret": None}


def test_hung_worker_is_killed_and_direct_child_reaped(tmp_path):
    pidfile = tmp_path / "pid"
    start = time.monotonic()
    with pytest.raises(net.SupervisionFailure, match="hard deadline"):
        run(f'import os,time; open({str(pidfile)!r},"w").write(str(os.getpid())); time.sleep(20)', timeout=1)
    assert time.monotonic() - start < 1.5
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)


def test_descendants_are_group_killed_on_success(tmp_path):
    childpid = tmp_path / "child"
    code = ('import subprocess,sys,json; p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(20)"],'
            'stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); '
            f'open({str(childpid)!r},"w").write(str(p.pid)); print(json.dumps({{"ok":True}}))')
    assert run(code) == {"ok": True}
    pid = int(childpid.read_text())
    # On Linux the subreaper directly reaps the adopted process; on other POSIX
    # kernels the orphan may briefly be a zombie awaiting the system reaper.
    if sys.platform.startswith("linux"):
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    else:
        import subprocess
        result = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True)
        assert not result.stdout.strip() or result.stdout.strip().startswith("Z")


def test_pipe_holding_descendant_cannot_extend_timeout():
    with pytest.raises(net.SupervisionFailure, match="hard deadline"):
        run('import subprocess,sys; subprocess.Popen([sys.executable,"-c","import time;time.sleep(20)"])', timeout=1)


@pytest.mark.parametrize("code,expected", [
    ('print("x"*5000)', "response bound"),
    ('print("not-json-secret")', "invalid JSON"),
    ('print("[]")', "invalid envelope"),
    ('import sys; print("credential-leak",file=sys.stderr); sys.exit(1)', "worker failed"),
])
def test_bounded_sanitized_failure(code, expected):
    with pytest.raises(net.SupervisionFailure, match=expected) as error:
        run(code, max_output_bytes=1024)
    assert "credential" not in str(error.value)
    assert "not-json-secret" not in str(error.value)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True, .1])
def test_bad_or_insufficient_timeout(timeout):
    with pytest.raises(net.SupervisionFailure):
        net.supervise([sys.executable, "-c", "raise SystemExit('must not run')"], {}, timeout=timeout,
            deadline=datetime.now(timezone.utc) + timedelta(seconds=3))


def test_past_provider_deadline_rejects_before_launch():
    with pytest.raises(net.SupervisionFailure, match="Insufficient time"):
        net.supervise([sys.executable, "-c", "pass"], {}, timeout=30,
            deadline=datetime.now(timezone.utc) - timedelta(seconds=1))


def test_request_size_bound():
    with pytest.raises(net.SupervisionFailure, match="input bound"):
        net.supervise([sys.executable, "-c", "pass"], {"payload": "x" * 65536}, timeout=2,
            deadline=datetime.now(timezone.utc) + timedelta(seconds=3))
