from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
RUNNER = ROOT / "scripts/run_dev40.py"
STOPPER = ROOT / "scripts/stop_at.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("qwen_dev40_test_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader
    spec.loader.exec_module(module)
    return module


RUN = load_runner()


def load_stopper():
    spec = importlib.util.spec_from_file_location("qwen_dev40_test_stopper", STOPPER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader
    spec.loader.exec_module(module)
    return module


STOP = load_stopper()


def write600(path: Path, value, *, json_value=True):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if json_value:
        path.write_text(json.dumps(value))
    else:
        path.write_text(value)
    path.chmod(0o600)


class FakeResponse:
    status = 200

    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, _limit):
        return self.body


class Development40Tests(unittest.TestCase):
    def test_plan_is_disabled_development_only_and_source_bound(self):
        plan = json.loads((ROOT / "configs/execution_plan.json").read_text())
        self.assertFalse(plan["execution_enabled"])
        self.assertEqual((plan["reports"], plan["passes"], plan["maximum_generations"]), (40, 1, 40))
        self.assertFalse(plan["validation_inference"] or plan["training"] or plan["bulk_extraction"])
        self.assertFalse(plan["organizer_labels_in_model_inputs"])
        self.assertEqual(set(plan["source_sha256"]), RUN.REQUIRED_SOURCES)
        for relative, expected in plan["source_sha256"].items():
            actual = hashlib.sha256((REPO / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, expected)

    def test_exact_model_prompt_and_limits_are_frozen(self):
        plan = json.loads((ROOT / "configs/execution_plan.json").read_text())
        self.assertEqual(plan["model_id"], "Qwen/Qwen3-14B")
        self.assertEqual(plan["model_revision"], "40c069824f4251a91eefaf281ebe4c544efd3e18")
        self.assertEqual(plan["prompt_sha256"], RUN.CONTROL_SHA)
        self.assertEqual(plan["definitions_sha256"], RUN.DEFINITIONS_SHA)
        self.assertEqual((plan["precision"], plan["attention"], plan["batch_size"]),
                         ("bfloat16", "sdpa", 1))
        self.assertEqual((plan["max_input_tokens"], plan["max_new_tokens"], plan["context_limit"]),
                         (8192, 2048, 40960))
        self.assertEqual(plan["stop_token_ids"], [151645, 151643])
        self.assertEqual((plan["automatic_retries"], plan["repair_generations"]), (0, 0))

    def test_validate_plan_rejects_wrong_hash(self):
        with self.assertRaises(RUN.GateError):
            RUN.validate_plan("0" * 64)

    def test_private_outputs_cannot_be_written_into_public_analysis(self):
        with self.assertRaises(RUN.GateError):
            RUN.private(REPO / "analysis/qwen_development40_v1/leak.json")

    def test_private_prepared_package_when_present(self):
        prepared = REPO / "state/runs/qwen-development40-v1-private/prepared-control-a"
        if not prepared.exists():
            self.skipTest("private prepared package is not part of the public checkout")
        plan_path = ROOT / "configs/execution_plan.json"
        plan = RUN.validate_plan(hashlib.sha256(plan_path.read_bytes()).hexdigest())
        rows, prompts, tokens = RUN.load_package(prepared, plan)
        self.assertEqual((len(rows), len(prompts), len(tokens)), (40, 40, 40))
        self.assertTrue(all(set(row) == {"case_index", "StudyInstanceUID", "Report", "split", "report_sha256"}
                            for row in rows))

    def test_real_development_authorization_and_watchdog_process_path(self):
        plan_sha = hashlib.sha256((ROOT / "configs/execution_plan.json").read_bytes()).hexdigest()
        plan = RUN.validate_plan(plan_sha)
        now = datetime.now(timezone.utc)
        pod_id = "synthetic-development40-pod"
        session = {
            "schema_version": 1,
            "approved_by_user": True,
            "execution_plan_sha256": plan_sha,
            "prepared_plan_sha256": RUN.PREPARED_PLAN_SHA,
            "prompt_sha256": RUN.CONTROL_SHA,
            "pod_id": pod_id,
            "region": "TEST-REGION-1",
            "gpu_name": "NVIDIA A40",
            "gpu_count": 1,
            "cloud_type": "SECURE",
            "container_disk_gb": 80,
            "persistent_volume_gb": 0,
            "network_volume_id": None,
            "compute_usd_per_hour": 0.49,
            "storage_usd_per_hour": 0.011,
            "maximum_usd": 3.0,
            "t0": (now - timedelta(seconds=30)).isoformat(),
            "hard_deadline": (now + timedelta(hours=2)).isoformat(),
            "shutdown_at": (now + timedelta(hours=1, minutes=55)).isoformat(),
        }
        observation = {
            "pod_id": pod_id,
            "provider_state": "RUNNING",
            "observed_at": now.isoformat(),
            "region": session["region"],
            "gpu_name": session["gpu_name"],
            "gpu_count": 1,
            "cloud_type": "SECURE",
            "container_disk_gb": 80,
            "persistent_volume_gb": 0,
            "network_volume_id": None,
            "actual_compute_usd_per_hour": 0.49,
            "actual_storage_usd_per_hour": 0.011,
            "current_total_usd_per_hour": 0.501,
        }
        with tempfile.TemporaryDirectory(prefix="qwen-dev40-watchdog-") as temporary:
            directory = Path(temporary)
            directory.chmod(0o700)
            session_path, key_path = directory / "session.json", directory / "runpod.key"
            observation_path = directory / "observation.json"
            receipt, result = directory / "receipt.json", directory / "result.json"
            cancel = directory / "cancel.json"
            write600(session_path, session)
            write600(observation_path, observation)
            write600(key_path, "synthetic_key_for_zero_network_calls", json_value=False)
            command = [sys.executable, str(STOPPER), "--arm", "--session", str(session_path),
                "--key", str(key_path), "--pod-id", pod_id, "--plan-sha256", plan_sha,
                "--result", str(result), "--cancel-marker", str(cancel), "--receipt", str(receipt)]
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
            stop = json.loads(receipt.read_text())
            pid = stop["watchdog_pid"]
            try:
                deadline = time.monotonic() + 5
                while True:
                    try:
                        room = RUN.validate_session(session, observation, stop, plan_sha,
                            RUN.sha(session_path), plan, session_path)
                        break
                    except RUN.GateError as exc:
                        if "not live" not in str(exc) or time.monotonic() >= deadline:
                            raise
                        time.sleep(0.05)
                self.assertEqual(room, plan["hard_inference_seconds"] - 1.0)

                old_ab_plan = REPO / "analysis/qwen_evidence_selection_v1/configs/execution_plan.json"
                with self.assertRaises(STOP.GateError):
                    STOP.validate_intent(session, RUN.sha(old_ab_plan), pod_id, now=now)
                with self.assertRaises(STOP.GateError):
                    STOP.validate_intent(session, plan_sha, "wrong-pod", now=now)

                changed = dict(session)
                changed["t0"] = (now - timedelta(seconds=31)).isoformat()
                changed_path = directory / "changed-session.json"
                write600(changed_path, changed)
                with self.assertRaises(RUN.GateError):
                    RUN.validate_session(changed, observation, stop, plan_sha,
                        RUN.sha(changed_path), plan, changed_path)

                calls = []
                body = json.dumps({"data": {"podStop": {
                    "id": pod_id, "desiredStatus": "EXITED"
                }}}).encode()

                def fake_opener(request, timeout):
                    calls.append((request.full_url, timeout))
                    return FakeResponse(body)

                stopped = STOP.stop_once(pod_id, "synthetic_key_for_zero_network_calls",
                                         opener=fake_opener)
                self.assertEqual(stopped["pod_id"], pod_id)
                self.assertEqual(calls, [(STOP.API, 20)])
            finally:
                try:
                    os.killpg(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    unittest.main()
