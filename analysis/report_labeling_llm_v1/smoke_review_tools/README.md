# Private smoke review generation

These postprocessing tools summarize the already-recorded five-study development smoke. They do not run models, change prompts, remove code fences, or evaluate validation labels. The inference code/config/prompt fingerprints remain those recorded by the executed revision. The summary records its own postprocessing script hash separately.

From the repository root, after copying the four complete run directories into private state:

```bash
python analysis/report_labeling_llm_v1/smoke_review_tools/summarize_smoke.py \
  --repo "$PWD" \
  --runs "$RSNA_STATE_DIR/runs/report-labeling-llm-smoke-20260914/remote_snapshot" \
  --output "$RSNA_STATE_DIR/runs/report-labeling-llm-smoke-20260914/smoke_review"
python analysis/report_labeling_llm_v1/smoke_review_tools/build_review.py \
  --review "$RSNA_STATE_DIR/runs/report-labeling-llm-smoke-20260914/smoke_review" \
  --template analysis/report_labeling_llm_v1/smoke_review_tools/case_viewer.template.html
```

Set `RSNA_STATE_DIR` to the restored private state directory, or pass `--state-dir` to the first command. The default is the checkout's ignored `state/` directory. Source metadata and the original prepared inputs must be restored there. Both model repeats must be complete and fingerprint-valid; missing runs fail rather than being assigned invented scores. The output directory must not already exist.

The second command needs Matplotlib and NumPy, already available in the local analysis environment. It writes `SMOKE_SUMMARY.md`, `smoke_metrics.csv`, `smoke_case_comparison.csv`, `smoke_dashboard.png`, `smoke_dashboard.pdf`, and `case_viewer.html`. Detailed reports, prompts, raw responses and reference labels remain in this private output directory. Only intentionally selected aggregate results belong in Git.

The dashboard uses 60 condition cells per run, with five distinct studies. Repeatability separately shows identical outputs including failures and outputs that are both valid and identical. Accepted binary agreement with organizer labels is descriptive development evidence, not validated accuracy or permission to scale.

The summarizer also writes `language.csv` and `agreement.csv` from the same five cases. Agreement uses the first repeat, with separate primary/secondary variants; an intersection requires every member to have a valid binary label and identical sign. Technical failures and semantic abstentions cannot vote. Conditional agreement is undefined when no cells are accepted.

Optional offline format inspection, with no model call or change to recorded acceptance:

```bash
python analysis/report_labeling_llm_v1/smoke_review_tools/diagnose_format.py \
  --review "$RSNA_STATE_DIR/runs/report-labeling-llm-smoke-20260914/smoke_review"
```

It writes a private per-case format CSV and a safe aggregate JSON. Searching for the presence of a complete condition object is diagnostic only: it never extracts that object into benchmark predictions. Outer-fence removal is tested only as post hoc syntactic salvageability. Manual error annotations and the v2 proposal in the saved private package are analyst review, not automatically generated ground truth.

The preserved final local package is named `smoke_review_v2/`: this suffix versions postprocessing outputs only, **not** a new model experiment. The original `smoke_review/` is retained. To regenerate, choose a fresh output directory and pass that same directory to the rendering and diagnostic commands.
