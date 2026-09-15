# Bounded Qwen runner and blinded review

The [Qwen comparison design](../qwen_report_extraction_v1/README.md) now has a separately versioned runtime and review workflow. **Actual Qwen inference remains NOT RUN.** All completed executions in this package use explicitly fabricated CPU fixtures. [Protocol](PROTOCOL.md) · [Results and limitations](RESULTS.md).

The runtime runs four supervised workers in control/candidate/candidate/control order, saves attempted-call and raw-output records before parsing, verifies both parser outputs, and stops after an incomplete generation. The complete-session technical summary tests all60 condition cells for validity and repeat identity under the primary v2 parser. Confidence, evidence spans and normalization metadata are part of that comparison. The strict v1 parser is a separate sensitivity result.

The review interface shows report text and extraction side by side, with random output codes hiding prompt arm/repeat. Organizer answers and the historical reference-containing review files are excluded. Blank/incomplete review cannot be finalized; no tool certifies reviewer qualifications. No model score or clinical improvement is measured by a synthetic demo.

## Local commands

From the repository root, with the previous private Qwen audit restored:

```bash
export RSNA_STATE_DIR=/path/to/private/state
python analysis/qwen_runtime_v1/scripts/runtime_qwen.py prepare \
  --state "$RSNA_STATE_DIR" \
  --candidate "$RSNA_STATE_DIR/runs/qwen-report-extraction-v1-20260915-reviewed" \
  --prepared "$RSNA_STATE_DIR/runs/qwen-runtime-NEW"
```

Preparation checks the fixed audit receipt, real-input digest and source/split fingerprints. It copies only inputs, prompt files, pinned tokenization, split metadata and the audit receipt. It never reads the historical details/review CSV or validation reports. Restoring the exact private audit is necessary: GitHub contains code and safe aggregates, not the dataset or private receipts. All output directories must be new.

To exercise the supervisor and review with fabricated input only:

```bash
python analysis/qwen_runtime_v1/scripts/runtime_qwen.py demo \
  --prepared "$RSNA_STATE_DIR/runs/qwen-demo-NEW"
python analysis/qwen_runtime_v1/scripts/review_qwen.py bundle \
  --prepared "$RSNA_STATE_DIR/runs/qwen-demo-NEW" \
  --output "$RSNA_STATE_DIR/runs/qwen-demo-review-NEW" --allow-synthetic-demo
python -m http.server 8800 --bind 127.0.0.1 \
  --directory "$RSNA_STATE_DIR/runs/qwen-demo-review-NEW/blinded"
```

Open [local review](http://127.0.0.1:8800/) while the server is running. Serve **only blinded/**; the sibling operator directory is private. Demo banners identify fabricated inputs and outputs. Browser selections live in page memory; download the JSON before closing and use the import control to resume. Never submit fabricated review as clinical adjudication.

For recovery accounting after a failed or interrupted session:

```bash
python analysis/qwen_runtime_v1/scripts/runtime_qwen.py inspect \
  --prepared "$RSNA_STATE_DIR/runs/qwen-demo-NEW"
```

This counts durable attempt-start intents, raw records and parsed records, and flags partial JSONL tails. An intent may be recorded before a model call actually begins. It is diagnostic accounting, not verification of an incomplete experiment's predictions. Completed-session verification remains mandatory for review bundles.

The `run --execute` CLI and direct backend helper both refuse GPU loading in this version, even if policy flags are changed. The offline Qwen adapter is wired for a future separately reviewed execution version. No credential, resource grant, provider shutdown timer or paid deployment is created here. [Runtime settings](configs/runtime.json) document the limits.

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_qwen_runtime.py
```

The new tests cover worker provenance, limits/failures, raw durability, parser integrity, primary repeat gates, non-overwrite rules, private/blinded boundaries and review collection. Source reuse is documented in [REVIEW.md](REVIEW.md); every historical source remains byte-identical.
