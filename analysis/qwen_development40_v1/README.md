# Qwen development benchmark of baseline A

This separately versioned runner evaluates the frozen control-A extractor on all
40 studies in the existing development split. It does not modify the completed
five-report A/B experiment.

The model input package contains only each development report, its study key,
split marker and source hash. Organizer labels are excluded from prompts and from
the runner. Evaluation may join the saved outputs to development labels only
after all 40 raw and parsed records are durable. The 18-study validation split is
outside this stage.

Scientific settings remain Qwen/Qwen3-14B revision
`40c069824f4251a91eefaf281ebe4c544efd3e18`, BF16, SDPA, greedy,
non-thinking, batch size 1, 8,192 input tokens, 2,048 output tokens and the
unchanged primary-v2/secondary-v1 parsers. There are no retries or repairs.

The private prepared package is:

```text
state/runs/qwen-development40-v1-private/prepared-control-a
```

Its plan SHA-256 is
`b2c6f7e4d220ab83f5556170334fa441dac28c495e3ab9e9b8fefc8e956ca023`.
All 40 inputs pass token limits; input counts range from 992 to 2,600 tokens.

The execution-plan SHA-256 is
`4af4d19ecfdd612ce96ab04c9760f1fb874b6506dd2df5e368ef4c8ccf940b52`.
Execution is disabled by default.

## Local verification

```bash
PY=state/runs/qwen-evidence-selection-v1-private/tokenizer-venv/bin/python
PLAN=4af4d19ecfdd612ce96ab04c9760f1fb874b6506dd2df5e368ef4c8ccf940b52
PREP=state/runs/qwen-development40-v1-private/prepared-control-a
CACHE=state/cache/tokenizer-preflight

"$PY" analysis/qwen_development40_v1/scripts/run_dev40.py \
  --prepared "$PREP" --cache "$CACHE" --plan-sha256 "$PLAN"

"$PY" analysis/qwen_development40_v1/scripts/run_dev40.py \
  --prepared "$PREP" --cache "$CACHE" --plan-sha256 "$PLAN" \
  --rehearsal --output state/runs/qwen-development40-v1-private/NEW-synthetic-output
```

The first command performs a zero-generation token-parity dry run. The second
performs 40 synthetic calls through the same supervisor, durable output and
parser dispatch used by the real path.

## Real path

`--run` additionally requires fresh private 0600 files for user approval, a
recent exact-pod `RUNNING` observation and a live pod-local shutdown receipt.
The approval must bind the exact execution plan, prepared plan, prompt, pod,
region, one allowlisted 48 GB Secure Cloud GPU, 80 GB container disk, no
persistent/network volume, signed-in rates, at most three hours and at most
$3. The runner rechecks watchdog liveness before every generation and uses a
hard supervised inference timeout below 1,800 seconds with a 15-minute
copy/stop reserve.

The shutdown receipt must be created by this package's own entry point after
the private development-40 session file exists:

```bash
"$PY" analysis/qwen_development40_v1/scripts/stop_at.py --arm \
  --session PRIVATE_SESSION_JSON --key PRIVATE_RUNPOD_KEY \
  --pod-id EXACT_APPROVED_POD_ID --plan-sha256 "$PLAN" \
  --result PRIVATE_STOP_RESULT_JSON --cancel-marker PRIVATE_CANCEL_JSON \
  --receipt PRIVATE_SHUTDOWN_RECEIPT_JSON
```

The receipt binds this execution plan, the frozen prepared plan, the exact
session file hash, pod ID, hard deadline and shutdown time. The runner verifies
the watchdog PID and exact command before model loading and before every
generation. The historical A/B watchdog is not accepted for this benchmark.

Based on the completed five-report A/B session, generation averaged about
32 seconds per report; 40 reports therefore have an inference-only estimate of
about 22 minutes. Model loading, cache preparation, copying and provider
conditions are additional and must be measured in an approved live session.

## Completed live benchmark

The approved baseline-A development benchmark completed on **2026-09-26**:
40/40 generations, 480 condition rows, 442 technically valid rows, 21
`ambiguous_evidence_error` rows and 17 `evidence_error` rows. Generation took
1,142.35 seconds and peaked at 28.29 GiB allocated GPU memory. The private
archive was copied locally and its archive plus internal artifact hashes were
verified before the pod was stopped. The signed-in console then showed the pod
not running at $0.00/hour, and the temporary control key was disabled.

See [the aggregate results](results/2026-09-26_181548_SGT/RESULTS.md) and the
[reproducible operator SOP](results/2026-09-26_181548_SGT/SOP.md). Raw reports,
study identifiers, model responses, tokens and private receipts remain under
ignored `state/`; they are not in Git.
