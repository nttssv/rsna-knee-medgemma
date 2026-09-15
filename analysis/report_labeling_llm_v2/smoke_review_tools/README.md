# Local visual review tooling

Prepare the private five-case package first, then choose a new output directory inside it:

```bash
python analysis/report_labeling_llm_v2/scripts/smoke_review.py \
  --prepared state/runs/report-labeling-llm-v2-local-NEW \
  --output state/runs/report-labeling-llm-v2-local-NEW/smoke_review
```

Without `--runs`, every v2 measurement is **NOT RUN**. The private viewer shows the original five report inputs, organizer references and frozen rule output, with no synthetic model predictions. The dashboard's ten panels have no invented numeric bars. Full reports stay in private state.

A future adapter may supply `--runs PREPARED/adapter_runs` containing `medgemma-1`, `medgemma-2`, `qwen-1`, `qwen-2`. Each must contain a completed `run_manifest.json` with `model_key`, `model_revision`, `prepared_manifest_sha256`, `candidate_code_sha256`, `predictions_sha256`, and optional `peak_gpu_allocated_gib`. Each `predictions.jsonl` must have the exact five StudyInstanceUIDs in order, `raw_output`, `generation_status`, `response` (the unchanged return from v2 validate_response), `rendered_prompt`, `runtime_seconds`, `input_tokens`, `output_tokens`. The tool verifies hashes/revisions, rejects partial or mismatched runs, and checks saved parsing against the current v2 contract. It never accepts v1 files as v2 runs.

Prepared manifests become stale when protocol/code/config/prompt/tests/templates change. Prepare a fresh directory before future execution. The [locked local adapter](../RUNTIME.md) implements this artifact interface; real CUDA integration remains unverified. Synthetic backend artifacts are rejected, and adapter-version-1 manifests also verify runtime artifact hashes. A runtime review must bind the rendered template and generated outputs to the pinned execution recipe.

Outputs are `SMOKE_SUMMARY.md`, `smoke_metrics.csv`, `smoke_case_comparison.csv`, `smoke_format_diagnostics.csv`, `smoke_semantic_review.csv`, `smoke_dashboard.png`, `smoke_dashboard.pdf`, and `case_viewer.html`. The future semantic CSV starts unreviewed; never treat its existence as completed adjudication. Missing GPU-memory measurements remain NOT MEASURED. Evidence buttons highlight exact original source offsets, including non-BMP Unicode characters.

Run the script in the existing CPU environment with Matplotlib/NumPy, or install repository dependencies. All outputs must remain private unless separately reviewed for aggregate-only publication.

To construct a future human approval artifact, a qualified reviewer completes the CSV for the selected model’s first run and preserves `candidate_code_sha256` from the prepared manifest alongside `review_source: human` and the 60 reviewed rows. The gate checks both experiment fingerprints and row hashes. A changed repeat blocks advancement even when the first run has been reviewed. This tooling does not create an approved human artifact automatically.

Measured rendering now also requires `session_start.json`, the reviewed plan, four ordered parent dispatches and a completed `session_manifest.json` with every child manifest hash. Child plan/session IDs, order and dispatch hashes must all agree. Standalone, reordered or mixed-session results are rejected before metrics are displayed.
