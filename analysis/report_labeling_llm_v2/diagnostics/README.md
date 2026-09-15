# Private partial-run diagnostic

This standalone reader supports only the recorded failed RTX smoke. It is outside the executed candidate's hashed source folders and does not weaken the existing completed-session viewer or gate.

After restoring private state, from the repository root:

```bash
python analysis/report_labeling_llm_v2/diagnostics/build_partial_review.py \
  --prepared state/runs/report-labeling-llm-v2-ada-executed-20260915 \
  --output state/runs/report-labeling-llm-v2-ada-executed-20260915/partial_review-NEW
python -m http.server 8795 --bind 127.0.0.1 \
  --directory state/runs/report-labeling-llm-v2-ada-executed-20260915/partial_review-NEW
```

Open `http://127.0.0.1:8795/case_viewer.html`. Choose a report and run. Original reports retain their language; the exact rendered prompt is expandable for attempted cases. All twelve organizer references appear beside frozen rule outputs and the model's technical status. Accuracy is undefined because no labels were accepted.

The reader checks all 17 original artifacts against the pinned remote inventory, current executed-code/preparation hashes, study order, and exact saved raw/parsed correspondence. It rejects changed or missing artifacts and refuses to overwrite a viewer directory. Public [aggregate results](../aggregate/rtx_smoke.json) contain no identifiers or reports. Rendered HTML contains private data: keep it in ignored state and serve on loopback only. This diagnostic cannot authorize or launch inference.
