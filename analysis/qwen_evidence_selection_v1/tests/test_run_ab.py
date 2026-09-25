"""CPU-only safety and completed private rehearsal checks; no provider calls."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_ab
import stop_at


class FakeResponse:
    status = 200
    def __enter__(self): return self
    def __exit__(self, *_): return False
    def read(self, _):
        return b'{"data":{"podStop":{"id":"synthetic-pod","desiredStatus":"EXITED"}}}'


class SupervisedAdapterTests(unittest.TestCase):
    def test_new_plan_and_science_bindings(self):
        plan = run_ab.validate_plan(run_ab.sha(run_ab.PLAN))
        self.assertEqual(plan["prepared_plan_sha256"], run_ab.PREPARED_SHA)
        self.assertEqual(plan["prompt_sha256"], {"A":run_ab.A_SHA, "B":run_ab.B_SHA,
                                                  "definitions":run_ab.DEFINITIONS_SHA})
        self.assertEqual(plan["default_action"], "dry-run")
        self.assertEqual(plan["maximum_generations"], 20)

    def test_effective_backend_recipe_drift_fails_before_model_load(self):
        plan = run_ab.read(run_ab.PLAN)
        source = run_ab.read(REPO / "analysis/qwen_report_extraction_v1/configs/experiment.json")
        run_ab.verify_effective_recipe(source, plan)
        for change in ({"seed": source["seed"] + 1},
                       {"max_new_tokens": source["max_new_tokens"] + 1},
                       {"revision": "different-revision"}):
            with self.subTest(change=change), self.assertRaises(run_ab.GateError):
                run_ab.verify_effective_recipe(dict(source, **change), plan)

    def synthetic_session(self):
        now = datetime.now(timezone.utc)
        t0 = now - timedelta(minutes=1)
        deadline = t0 + timedelta(hours=2)
        stop_at_time = deadline - timedelta(minutes=5)
        plan_sha = run_ab.sha(run_ab.PLAN)
        session = {"schema_version":1,"approved_by_user":True,"execution_plan_sha256":plan_sha,
            "prepared_plan_sha256":run_ab.PREPARED_SHA,"prompt_sha256":run_ab.read(run_ab.PLAN)["prompt_sha256"],
            "pod_id":"synthetic-pod","region":"SYNTHETIC-REGION","gpu_name":"NVIDIA RTX A6000",
            "gpu_count":1,"cloud_type":"SECURE","container_disk_gb":80,"persistent_volume_gb":0,
            "network_volume_id":None,"compute_usd_per_hour":0.53,"storage_usd_per_hour":0.011,
            "maximum_usd":3.0,"t0":t0.isoformat(),"hard_deadline":deadline.isoformat(),
            "shutdown_at":stop_at_time.isoformat()}
        observed = {"pod_id":"synthetic-pod","provider_state":"RUNNING","observed_at":now.isoformat(),
            "region":session["region"],"gpu_name":session["gpu_name"],"gpu_count":1,"cloud_type":"SECURE",
            "container_disk_gb":80,"persistent_volume_gb":0,"network_volume_id":None,
            "actual_compute_usd_per_hour":0.53,"actual_storage_usd_per_hour":0.011,
            "current_total_usd_per_hour":0.541}
        stop = {"pod_id":"synthetic-pod","execution_plan_sha256":plan_sha,"session_sha256":"0"*64,
                "shutdown_at":session["shutdown_at"],"checked_at":now.isoformat(),
                "watchdog_pid":99999999,"watchdog_argv":[sys.executable,
                    str(ROOT/"scripts/stop_at.py"),"--pod-id","synthetic-pod",
                    "--plan-sha256",plan_sha]}
        return session, observed, stop, plan_sha, now

    def test_fresh_grant_and_exact_observation_reject_drift(self):
        session, observed, stop, plan_sha, now = self.synthetic_session()
        for changed, where in [({"approved_by_user":False}, "session"),
                               ({"gpu_name":"NVIDIA H100"}, "session"),
                               ({"gpu_count":2}, "observation"),
                               ({"cloud_type":"COMMUNITY"}, "observation"),
                               ({"actual_compute_usd_per_hour":0.8}, "observation"),
                               ({"provider_state":"STOPPED"}, "observation"),
                               ({"session_sha256":"1"*64}, "stop")]:
            s,o,w = dict(session),dict(observed),dict(stop)
            {"session":s,"observation":o,"stop":w}[where].update(changed)
            with self.subTest(changed=changed), self.assertRaises(run_ab.GateError):
                run_ab.validate_session(s,o,w,plan_sha,"0"*64,now=now)

    def test_external_stop_request_is_exact_pod_and_once(self):
        calls=[]
        def opener(request, timeout):
            calls.append((request.full_url, request.data, timeout))
            return FakeResponse()
        result=stop_at.stop_once("synthetic-pod", "synthetickey0123456789012345", opener=opener)
        self.assertEqual(result["status"], "stop_request_accepted")
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][0],stop_at.API)
        self.assertIn(b'podId: \\"synthetic-pod\\"',calls[0][1])

    def test_shutdown_worker_remains_eligible_after_deadline(self):
        session, _, _, plan_sha, _ = self.synthetic_session()
        late = run_ab.aware(session["hard_deadline"]) + timedelta(seconds=10)
        with self.assertRaises(run_ab.GateError):
            stop_at.validate_intent(session, plan_sha, "synthetic-pod", now=late)
        self.assertEqual(stop_at.validate_intent(session, plan_sha, "synthetic-pod",
                                                 worker_due=True, now=late),
                         run_ab.aware(session["shutdown_at"]))

    def test_private_rehearsal_is_exact_order_and_marked_synthetic(self):
        private = REPO / "state/runs/qwen-evidence-selection-v1-private"
        prepared = private / "prepared-local-final2"
        output = private / "rehearsal-adapter-normal"
        if not (prepared / "plan.json").is_file() or not (output / "session_result.json").is_file():
            self.skipTest("Private reports/rehearsal intentionally absent from a public checkout")
        plan=run_ab.validate_plan(run_ab.sha(run_ab.PLAN))
        rows,prompts,tokens=run_ab.load_package(prepared,
            REPO / "state/runs/qwen-runtime-v1-20260915-prepared-final",plan)
        self.assertEqual(len(rows),5)
        summary=json.loads((output/"session_result.json").read_text())
        self.assertEqual([summary[k] for k in ("planned","attempted","completed","failed","unrun")],
                         [20,20,20,0,0])
        self.assertTrue(summary["synthetic"])
        self.assertEqual(summary["blocks_completed"],list(run_ab.ORDER))
        seen=[]
        for block in run_ab.ORDER:
            arm=block[0]
            attempts=run_ab.read_lines(output/block/"attempts.jsonl")
            raw=run_ab.read_lines(output/block/"raw.jsonl")
            parsed=run_ab.read_lines(output/block/"parsed.jsonl")
            self.assertEqual((len(attempts),len(raw),len(parsed)),(5,5,5))
            self.assertTrue(all(r["synthetic"] and r["prompt_sha256"]==plan["prompt_sha256"][arm]
                                and r["input_ids"]==tokens[arm][i]["input_ids"]
                                and r["rendered_prompt"]==tokens[arm][i]["rendered_prompt"]
                                for i,r in enumerate(raw)))
            self.assertTrue(all(p["synthetic"] and len(p["primary_v2"]["rows"])==12 for p in parsed))
            seen.extend(a["attempt"] for a in attempts)
        self.assertEqual(seen,list(range(1,21)))
        with tempfile.TemporaryDirectory(dir=private) as temp:
            altered=Path(temp)/"altered"
            shutil.copytree(prepared,altered)
            target=altered/"tokenized.json"
            target.write_text(target.read_text().replace('"input_tokens": 1298','"input_tokens": 1299',1))
            with self.assertRaises(run_ab.GateError):
                run_ab.load_package(altered,REPO/"state/runs/qwen-runtime-v1-20260915-prepared-final",plan)

    def test_fresh_supervised_cpu_dispatch_and_failures(self):
        private = REPO / "state/runs/qwen-evidence-selection-v1-private"
        prepared = private / "prepared-local-final2"
        python = private / "tokenizer-venv/bin/python"
        source = REPO / "state/runs/qwen-runtime-v1-20260915-prepared-final"
        cache = REPO / "state/cache/tokenizer-preflight"
        if not all(p.exists() for p in (prepared / "plan.json", python, source / "plan.json", cache)):
            self.skipTest("Private five-report package and tokenizer are absent from public checkout")
        with tempfile.TemporaryDirectory(dir=private) as temp:
            base = [str(python), str(ROOT / "scripts/run_ab.py"), "--rehearsal",
                "--prepared", str(prepared), "--source-prepared", str(source),
                "--cache", str(cache), "--plan-sha256", run_ab.sha(run_ab.PLAN)]
            normal = Path(temp) / "normal"
            first = subprocess.run(base + ["--output", str(normal)], capture_output=True, text=True, timeout=30)
            self.assertEqual(first.returncode, 0, first.stderr)
            summary = run_ab.read(normal / "session_result.json")
            self.assertEqual([summary[k] for k in ("planned", "attempted", "completed", "failed", "unrun")],
                             [20, 20, 20, 0, 0])
            self.assertEqual(summary["blocks_completed"], list(run_ab.ORDER))
            self.assertTrue(summary["synthetic"])
            self.assertEqual(sum(len(run_ab.read_lines(normal / b / "raw.jsonl")) for b in run_ab.ORDER), 20)
            again = subprocess.run(base + ["--output", str(normal)], capture_output=True, text=True, timeout=30)
            self.assertNotEqual(again.returncode, 0)  # No overwrite or repeat selection.
            incomplete = Path(temp) / "incomplete"
            failed = subprocess.run(base + ["--output", str(incomplete), "--fake-behavior", "incomplete"],
                                    capture_output=True, text=True, timeout=30)
            self.assertNotEqual(failed.returncode, 0)
            partial = run_ab.read(incomplete / "session_result.json")
            self.assertEqual([partial[k] for k in ("attempted", "completed", "failed", "unrun")],
                             [1, 0, 1, 19])
            self.assertEqual(len(run_ab.read_lines(incomplete / "A1/raw.jsonl")), 1)
            self.assertEqual(len(run_ab.read_lines(incomplete / "A1/technical_failures.jsonl")), 1)
            invalid = Path(temp) / "schema-invalid"
            schema = subprocess.run(base + ["--output", str(invalid), "--fake-behavior", "schema_error"],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(schema.returncode, 0, schema.stderr)
            self.assertEqual(sum(len(run_ab.read_lines(invalid / b / "technical_failures.jsonl"))
                                 for b in run_ab.ORDER), 20)
            timed = Path(temp) / "hard-timeout"
            hung = subprocess.run(base + ["--output", str(timed), "--fake-behavior", "hang",
                                          "--rehearsal-timeout", "2"],
                                  capture_output=True, text=True, timeout=15)
            self.assertNotEqual(hung.returncode, 0)
            receipt = run_ab.read(timed / "supervisor_result.json")
            self.assertEqual(receipt["status"], "hard_timeout")
            self.assertTrue(receipt["partial_artifacts_sha256"])
            for behavior in ("parsed_write_error_last", "block_receipt_error_last"):
                broken = Path(temp) / behavior
                run = subprocess.run(base + ["--output", str(broken), "--fake-behavior", behavior],
                                     capture_output=True, text=True, timeout=30)
                self.assertNotEqual(run.returncode, 0, run.stderr)
                outcome = run_ab.read(broken / "session_result.json")
                self.assertEqual(outcome["attempted"], 20)
                self.assertEqual(outcome["failed"], 0)  # Complete generation, failed durable finalization.
                self.assertEqual(outcome["failure_type"], "OSError")
                self.assertEqual(outcome["status"], "failed")
                self.assertEqual(run_ab.read(broken / "supervisor_result.json")["status"], "child_failed")
                if behavior == "parsed_write_error_last":
                    self.assertEqual(outcome["parsed_records_written"], 19)
                else:
                    self.assertEqual(outcome["parsed_records_written"], 20)
                    self.assertEqual(outcome["blocks_completed"], ["A1", "B1", "B2"])


if __name__ == "__main__":
    unittest.main()
