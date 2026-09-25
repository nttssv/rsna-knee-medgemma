# Qwen evidence-selection comparison v1

This prospective local experiment changes one prompt component: the evidence-selection workflow. A is a byte-for-byte copy of the successfully run control prompt. B changes only the evidence-selection guidance: review the complete report, match statements to the exact target, preserve explicit negation and decisive qualifiers, and quote one continuous supporting passage. The baseline snapshot records 20/60 raw proposal differences and 18/60 primary parsed-row differences between the representative control and candidate blocks. Target definitions, output schema, parsing, model, sampling and five-report inputs stay fixed.

The experiment is not proof that either prompt is medically correct. The five-report regression checklist is private and derived from previously observed development outputs; it is not independent validation or ground truth. Organizer labels are not opened. No report IDs, original report text, case-level output, raw model response or private checklist are published to GitHub.

Local preparation entry point:

```bash
state/runs/qwen-evidence-selection-v1-private/tokenizer-venv/bin/python analysis/qwen_evidence_selection_v1/scripts/prepare_local.py \
  --prepared state/runs/qwen-runtime-v1-20260915-prepared-final \
  --tokenizer-cache state/cache/tokenizer-preflight \
  --completed-outputs state/runs/qwen-extraction-a6000-20260925/final/outputs \
  --output state/runs/qwen-evidence-selection-v1-private/prepared-local-final
```

The entry point is offline-only, refuses to overwrite, validates the exact five development inputs and split, checks A against the prior prepared token IDs, and records B token counts. It does not load Qwen weights or access a GPU. The output remains in ignored `state/`.

`configs/experiment.json` freezes scientific settings and the A1 → B1 → B2 → A2 order. `scripts/run_ab.py` is the separately hashed A/B entry point. It validates the private prepared A/B package (SHA-256 `a26965fa84d7d2db37d8d3812f7b4061119df4bf77690127bb27d78e0f8eda82`), rebuilds both prompts from the same five reports, and checks every rendered prompt and token ID before dispatch. A1/A2 use the exact historical control; B1/B2 use the frozen `candidate_B.txt` (SHA-256 `f1b39d642bd4d7948be7d9460897641400968fe87657529e725507176a044ff1`). It does not call the old helper that reloads the historical candidate. The new [execution plan](configs/execution_plan.json) binds the adapter and reused backend, decoder, parsers, cache audit, effective Qwen configs and supervisor sources. It checks effective encoder settings before loading weights. Its SHA-256 is `6d551f7bf19db648e11eed9766e6a47b984567e6b8fbc6e4dd660c738bf9c54a`.

Default invocation is an offline **dry run** (zero generations). The same CLI has `--rehearsal` for a supervised 20-call **SYNTHETIC** CPU run, and `--run` for real offline Qwen generation through the frozen HF backend. Real execution additionally needs a fresh 0600 user approval, a fresh exact-pod RUNNING resource observation and a live external stop-process receipt; none is created or implied here. `execution_enabled` stays false in the scientific config, so the default action cannot rent or load a GPU. The real flag is gated by the separate private session, not a revision of the frozen scientific plan.

From the repository root, after placing the five-report package and pinned tokenizer cache in ignored private storage:

```bash
PY=state/runs/qwen-evidence-selection-v1-private/tokenizer-venv/bin/python
PLAN=6d551f7bf19db648e11eed9766e6a47b984567e6b8fbc6e4dd660c738bf9c54a
COMMON=(--prepared state/runs/qwen-evidence-selection-v1-private/prepared-local-final2 \
  --source-prepared state/runs/qwen-runtime-v1-20260915-prepared-final \
  --cache state/cache/tokenizer-preflight --plan-sha256 "$PLAN")
"$PY" analysis/qwen_evidence_selection_v1/scripts/run_ab.py "${COMMON[@]}"
"$PY" analysis/qwen_evidence_selection_v1/scripts/run_ab.py "${COMMON[@]}" \
  --rehearsal --output state/runs/qwen-evidence-selection-v1-private/NEW-synthetic-run
```

On a future separately approved GPU session, install the frozen pinned runtime dependencies and full pinned weight cache on the chosen pod, copy the private prepared/source packages there, and set `PY` and `COMMON` to those pod-local paths. Arm `scripts/stop_at.py` as a **separate process on that same pod** with the private owner-only session and RunPod control-key files before loading the model; only then can the inference worker independently check that watchdog PID through `/proc`. The operator must separately verify an **outside-pod** console/API stop route and supervise the session. The session clock must already include setup time. The operator copies and hashes outputs, stops early after success/failure and verifies provider stopped state and billing; a pod-local watchdog alone is not billing confirmation.

```bash
"$PY" analysis/qwen_evidence_selection_v1/scripts/stop_at.py --arm \
  --session /PRIVATE/session-0600.json --key /PRIVATE/runpod-key-0600 \
  --pod-id EXACT_APPROVED_POD_ID --plan-sha256 "$PLAN" \
  --receipt /PRIVATE/stop-receipt-0600.json --result /PRIVATE/stop-result-0600.json \
  --cancel-marker /PRIVATE/operator-verified-stopped.marker

"$PY" analysis/qwen_evidence_selection_v1/scripts/run_ab.py "${COMMON[@]}" --run \
  --session /PRIVATE/session-0600.json --observation /PRIVATE/fresh-running-observation-0600.json \
  --shutdown-receipt /PRIVATE/stop-receipt-0600.json --output /PRIVATE/NEW-qwen-output
```

The private session must bind this exact execution-plan SHA, prepared-plan SHA, A/B/definitions hashes, exact approved pod/region/one allowlisted GPU, Secure Cloud, 80 GB container disk, zero persistent and network volumes, actual signed-in compute/storage rates, a maximum of $3, T0, immutable hard deadline within 3 hours and `shutdown_at` at least 5 minutes before that deadline. The fresh RUNNING observation must establish the same resource and rates independently. The launcher requires room for the 1,800-second hard inference timeout plus a 15-minute copy/stop reserve; it has no retries, fallback pod, prompt repair or output overwrite. The full JSON field contract is enforced by `validate_session()`; do not fabricate either private receipt. No approval from the previous session applies to this one.
