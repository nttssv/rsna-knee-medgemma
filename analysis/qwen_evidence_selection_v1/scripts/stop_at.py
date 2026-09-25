"""Pod-local exact-pod RunPod stop backstop for one freshly approved A/B session."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
from run_ab import PLAN, GateError, aware, private, private_json, read, sha, validate_plan, write_new

API = "https://api.runpod.io/graphql"


def read_key(path: Path) -> str:
    path = private(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.getuid():
        raise GateError("Owner-only 0600 RunPod control key required")
    key = path.read_text().strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{20,256}", key):
        raise GateError("RunPod control key format is invalid")
    return key


def validate_intent(session: dict, plan_sha: str, pod_id: str, *, worker_due: bool = False,
                    now: datetime | None = None):
    validate_plan(plan_sha)
    if (session.get("approved_by_user") is not True or session.get("execution_plan_sha256") != plan_sha
        or session.get("pod_id") != pod_id or session.get("prepared_plan_sha256") != read(PLAN)["prepared_plan_sha256"]
        or not isinstance(pod_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", pod_id)):
        raise GateError("Shutdown intent does not bind the fresh approved exact pod and plan")
    t0, deadline, stop = map(aware, (session["t0"], session["hard_deadline"], session["shutdown_at"]))
    now = now or datetime.now(timezone.utc)
    rates = [session.get("compute_usd_per_hour"), session.get("storage_usd_per_hour")]
    ceiling = session.get("maximum_usd")
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in rates) \
            or type(ceiling) not in (int, float) or not math.isfinite(ceiling) or not 0 < ceiling <= 3 \
            or not 0 < (deadline - t0).total_seconds() <= 10800 \
            or sum(rates) * (deadline - t0).total_seconds() / 3600 > ceiling:
        raise GateError("External stop intent exceeds the approved session budget or duration")
    if (not worker_due and not t0 <= now < stop) or stop > deadline \
            or (deadline - stop).total_seconds() < 300:
        raise GateError("Approved shutdown deadline is invalid or expired")
    return stop


def stop_once(pod_id: str, key: str, *, opener=None) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", pod_id):
        raise GateError("Invalid exact pod ID")
    query = 'mutation { podStop(input: { podId: "' + pod_id + '" }) { id desiredStatus } }'
    request = urllib.request.Request(API, data=json.dumps({"query": query}).encode(), method="POST", headers={
        "Authorization": "Bearer " + key, "Accept": "application/json",
        "Content-Type": "application/json", "User-Agent": "rsna-qwen-evidence-selection/1.0"})
    if opener is None:
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                raise GateError("RunPod redirect refused")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect()).open
    try:
        with opener(request, timeout=20) as response:
            if response.status != 200:
                raise GateError("RunPod stop response was not successful")
            body = response.read(4096)
            if key.encode() in body:
                raise GateError("Provider echoed a credential")
            value = json.loads(body).get("data", {}).get("podStop", {})
            if value.get("id") != pod_id or value.get("desiredStatus") not in ("EXITED", "STOPPED"):
                raise GateError("RunPod stop response did not bind the exact pod")
            return {"status": "stop_request_accepted", "pod_id": pod_id,
                    "provider_desired_status": value["desiredStatus"],
                    "received_at": datetime.now(timezone.utc).isoformat()}
    except GateError:
        raise
    except Exception:
        # Unknown network outcome: never issue a second mutation automatically.
        raise GateError("Single stop request failed or its outcome is unknown") from None


def worker(args) -> int:
    session, key = private_json(args.session), read_key(args.key)
    stop_at = validate_intent(session, args.plan_sha256, args.pod_id, worker_due=True)
    result = {"pod_id": args.pod_id, "execution_plan_sha256": args.plan_sha256,
              "session_sha256": sha(args.session), "synthetic": False}
    while datetime.now(timezone.utc) < stop_at:
        if args.cancel_marker.exists():
            result.update(status="cancelled_after_operator_verified_stop")
            write_new(args.result, result)
            return 0
        time.sleep(min(10, max(0, (stop_at - datetime.now(timezone.utc)).total_seconds())))
    if args.cancel_marker.exists():
        result.update(status="cancelled_after_operator_verified_stop")
    else:
        try:
            result.update(stop_once(args.pod_id, key))
        except GateError:
            result.update(status="stop_request_failed_or_unknown")
    write_new(args.result, result)
    return 0 if result["status"] in ("stop_request_accepted", "cancelled_after_operator_verified_stop") else 1


def arm(args) -> int:
    session = private_json(args.session)
    validate_intent(session, args.plan_sha256, args.pod_id)
    read_key(args.key)
    for target in (args.receipt, args.result, args.cancel_marker):
        private(target)
        if target.exists():
            raise FileExistsError(f"Refuse to reuse shutdown artifact: {target}")
    args.receipt.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    log = args.receipt.with_suffix(".log")
    argv = [sys.executable, str(HERE), "--worker", "--session", str(args.session),
            "--key", str(args.key), "--pod-id", args.pod_id, "--plan-sha256", args.plan_sha256,
            "--result", str(args.result), "--cancel-marker", str(args.cancel_marker)]
    with log.open("xb") as stream:
        os.chmod(log, 0o600)
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    write_new(args.receipt, {"pod_id": args.pod_id, "execution_plan_sha256": args.plan_sha256,
        "session_sha256": sha(args.session),
        "shutdown_at": session["shutdown_at"], "checked_at": datetime.now(timezone.utc).isoformat(),
        "watchdog_pid": process.pid, "watchdog_argv": argv})
    print(json.dumps({"armed": True, "pod_id": args.pod_id, "watchdog_pid": process.pid,
                      "receipt": str(args.receipt)}))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--arm", action="store_true")
    actions.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    for flag in ("session", "key", "result", "cancel-marker", "receipt"):
        parser.add_argument("--" + flag, type=Path)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args(argv)
    required = (args.session, args.key, args.result, args.cancel_marker)
    if any(x is None for x in required) or (args.arm and args.receipt is None):
        parser.error("--session, --key, --result, --cancel-marker and (for --arm) --receipt are required")
    for flag in ("session", "key", "result", "cancel_marker", "receipt"):
        if getattr(args, flag) is not None:
            setattr(args, flag, private(getattr(args, flag)))
    return arm(args) if args.arm else worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
