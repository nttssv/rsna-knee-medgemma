"""Bound a single network worker using an external POSIX process-group deadline.

No shell, retry, inherited credential environment, or unbounded output is used.
The direct child is always reaped; Linux subreaper support also reaps adopted
descendants. Other POSIX kernels reap orphan descendants after group SIGKILL.
"""
from __future__ import annotations

import ctypes
from datetime import datetime, timezone
import json
import math
import os
import selectors
import signal
import subprocess
import sys
import time

MAX_OUTPUT_BYTES = 2 * 1024 * 1024
MAX_INPUT_BYTES = 65536
REAP_RESERVE_SECONDS = 0.25


class SupervisionFailure(RuntimeError):
    pass


def _linux_subreaper():
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
            raise SupervisionFailure("Cannot enable descendant reaping")


def _kill_and_reap(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    # Reaping a killed direct child cannot wait for a stuck CUDA/HTTP worker.
    try:
        process.wait(timeout=REAP_RESERVE_SECONDS)
    except subprocess.TimeoutExpired as exc:
        raise SupervisionFailure("Killed worker was not reaped in cleanup reserve") from exc
    if sys.platform.startswith("linux"):
        limit = time.monotonic() + REAP_RESERVE_SECONDS
        while time.monotonic() < limit:
            try:
                pid, _ = os.waitpid(-process.pid, os.WNOHANG)
                if pid == 0:
                    time.sleep(0.002)
                else:
                    continue
            except ChildProcessError:
                break


def supervise(command: list[str], payload: dict, *, timeout: float,
              deadline: datetime, max_output_bytes: int = MAX_OUTPUT_BYTES) -> dict:
    """Run one worker; all work plus bounded cleanup is inside the supplied limit.

    Payload goes through stdin, never argv/environment. Failure messages contain
    neither worker output nor credential paths. A deadline cannot be extended by
    a wall-clock rollback because the initial bound is also monotonic.
    """
    if os.name != "posix":
        raise SupervisionFailure("POSIX process groups are required")
    if (type(timeout) not in (int, float) or not math.isfinite(timeout)
            or timeout <= 0 or not isinstance(deadline, datetime) or deadline.tzinfo is None):
        raise SupervisionFailure("Finite timeout and aware deadline required")
    if type(max_output_bytes) is not int or not 1 <= max_output_bytes <= MAX_OUTPUT_BYTES:
        raise SupervisionFailure("Invalid output bound")
    now = datetime.now(timezone.utc)
    budget = min(float(timeout), (deadline - now).total_seconds())
    reserve = 2 * REAP_RESERVE_SECONDS
    if budget <= reserve:
        raise SupervisionFailure("Insufficient time for worker and cleanup reserve")
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
    if len(encoded) > MAX_INPUT_BYTES:
        raise SupervisionFailure("Worker request exceeds input bound")
    if not command or any(not isinstance(x, str) for x in command):
        raise SupervisionFailure("Explicit worker argv required")
    _linux_subreaper()
    cutoff = time.monotonic() + budget - reserve
    env = {"PATH": os.defpath, "LANG": "C.UTF-8", "PYTHONNOUSERSITE": "1"}
    selector = selectors.DefaultSelector()
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True, env=env)
    except Exception:
        selector.close()
        raise SupervisionFailure("Network worker could not start") from None
    buffer = bytearray()
    cursor = 0
    try:
        os.set_blocking(process.stdin.fileno(), False)
        os.set_blocking(process.stdout.fileno(), False)
        selector.register(process.stdin, selectors.EVENT_WRITE, "input")
        selector.register(process.stdout, selectors.EVENT_READ, "output")
        while selector.get_map():
            remaining = min(cutoff - time.monotonic(),
                (deadline - datetime.now(timezone.utc)).total_seconds() - reserve)
            if remaining <= 0:
                raise SupervisionFailure("Network worker exceeded hard deadline")
            for key, _ in selector.select(min(remaining, 0.05)):
                if key.data == "input":
                    try:
                        cursor += os.write(key.fd, encoded[cursor:])
                    except BrokenPipeError:
                        cursor = len(encoded)
                    if cursor == len(encoded):
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                else:
                    chunk = os.read(key.fd, min(65536, max_output_bytes + 1 - len(buffer)))
                    if not chunk:
                        selector.unregister(key.fileobj)
                    else:
                        buffer.extend(chunk)
                        if len(buffer) > max_output_bytes:
                            raise SupervisionFailure("Network worker exceeded response bound")
        remaining = cutoff - time.monotonic()
        if remaining <= 0:
            raise SupervisionFailure("Network worker exceeded hard deadline")
        try:
            code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            raise SupervisionFailure("Network worker exceeded hard deadline") from exc
        if code != 0:
            raise SupervisionFailure("Network worker failed")
        try:
            value = json.loads(buffer)
        except (ValueError, UnicodeDecodeError) as exc:
            raise SupervisionFailure("Network worker returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise SupervisionFailure("Network worker returned invalid envelope")
        return value
    finally:
        selector.close()
        _kill_and_reap(process)
        for stream in (process.stdin, process.stdout):
            if stream and not stream.closed:
                stream.close()
