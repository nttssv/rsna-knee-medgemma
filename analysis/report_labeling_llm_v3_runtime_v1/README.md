# MedGemma v3 · local runtime and blinded review

**GPU execution disabled. No v3 model results yet.** This milestone implements the bounded runner and two-stage review workflow for the unchanged [v3 candidate](../report_labeling_llm_v3/README.md).

| Step | Input | Output |
|---|---|---|
| Prepare locally | Frozen five-report development package, original 40/18 split, pinned candidate | New checksum-bound runtime plan |
| Future inference | Original-language report + exact target definitions + assigned prompt | Twelve four-state proposals, quoted evidence, confidence; exact raw tokens and receipts |
| First-stage review | Coded report and model proposal, organizer answers concealed | 240 human entailment judgments |
| Second-stage comparison | Validated first-stage review and unchanged organizer references | Separate agreement table and review-category counts |

The local **synthetic demo** walks through this software pipeline with fabricated text and fixed fabricated outputs. It loads no model. Read the [protocol](PROTOCOL.md) and [results/limitations](RESULTS.md).

## Prepare the real package without inference

From the repository root, with restored private state:

```bash
python analysis/report_labeling_llm_v3_runtime_v1/scripts/runtime_v3.py prepare \
  --candidate "$RSNA_STATE_DIR/runs/report-labeling-llm-v3-candidate-20260915" \
  --state "$RSNA_STATE_DIR" \
  --prepared "$RSNA_STATE_DIR/runs/report-labeling-llm-v3-runtime-prepared"
```

Use a new output name on every preparation. `run --execute` deliberately refuses in this revision. The old candidate's `NOT_EXECUTION_READY` record remains unchanged.

## Try the software demo on CPU

```bash
python analysis/report_labeling_llm_v3_runtime_v1/scripts/runtime_v3.py demo \
  --prepared "$RSNA_STATE_DIR/runs/v3-synthetic-demo"
python analysis/report_labeling_llm_v3_runtime_v1/scripts/review_v3.py bundle \
  --prepared "$RSNA_STATE_DIR/runs/v3-synthetic-demo" \
  --output "$RSNA_STATE_DIR/runs/v3-synthetic-review" --allow-synthetic-demo
python -m http.server 8797 --bind 127.0.0.1 \
  --directory "$RSNA_STATE_DIR/runs/v3-synthetic-review/blinded"
```

Open [local review demo](http://127.0.0.1:8797/). It shows the input report alongside the proposed output, unchanged definitions and human-review controls. Download your review JSON before closing; changes are held in page memory and no data is uploaded. Resume by importing that file. Serve **only `blinded/`**, never the parent directory or operator mapping.

After a complete review, `review_v3.py finalize --prepared ... --bundle ... --review ... --output ...` verifies all 240 judgments before producing the separate organizer comparison. A fabricated demo additionally requires `--allow-synthetic-demo`; it has no organizer answers. The software does not verify reviewer qualifications or grant permission to scale.

Code: [runner](scripts/runtime_v3.py), [review workflow](scripts/review_v3.py), [review interface](scripts/reviewer.html), [disabled policy](configs/runtime.json), [tests](tests/test_runtime_v3.py). Real model IDs/revisions and generation settings come from the frozen [candidate config](../report_labeling_llm_v3/configs/experiment.json). No weights, credentials, reports or study identifiers are included in Git.
