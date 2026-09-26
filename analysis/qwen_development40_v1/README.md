# Qwen development-40 benchmark v1

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
`4978baf4e91e2841676e22d2c6a0e68a8bffbdfa3b8651e4896161f626aa684b`.
Execution is disabled by default.

## Local verification

```bash
PY=state/runs/qwen-evidence-selection-v1-private/tokenizer-venv/bin/python
PLAN=4978baf4e91e2841676e22d2c6a0e68a8bffbdfa3b8651e4896161f626aa684b
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

Based on the completed five-report A/B session, generation averaged about
32 seconds per report; 40 reports therefore have an inference-only estimate of
about 22 minutes. Model loading, cache preparation, copying and provider
conditions are additional and must be measured in the approved live session.
No GPU execution or spending is authorized by this source milestone.
