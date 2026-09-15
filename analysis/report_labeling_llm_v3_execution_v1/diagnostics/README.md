# Measured execution dashboard

This CPU-only reader displays original-language reports, exact rendered prompts, raw responses and frozen parser results. It shows all four planned runs, including work that was not attempted, with separate technical and semantic denominators. It never displays organizer references or certifies medical correctness. Prompt arms are visible in this operator dashboard; a qualified semantic reviewer should use the separate blinded bundle before opening it.

The reader is outside the executed runner's fingerprinted source directories. It supports finalized complete or failed sessions without weakening the complete-session gate in the blinded review tool. Before reading, it requires the exact SHA-256 of the independently transferred remote file inventory, checks the complete file set and every byte checksum, validates the reviewed prepared plan and reparses every recorded response against its original report. Complete sessions additionally pass the original runtime verifier. The transfer inventory is an integrity receipt, not a provider signature or a substitute for the original run receipts.

From the repository root, after restoring private state and the separate transfer inventory:

```bash
python analysis/report_labeling_llm_v3_execution_v1/diagnostics/build_dashboard.py \
  --prepared "$RSNA_STATE_DIR/runs/report-labeling-llm-v3-execution-executed-20260915" \
  --inventory "$RSNA_STATE_DIR/runs/report-labeling-llm-v3-private-provenance-20260915/remote-file-hashes.json" \
  --inventory-sha256 19f31b50f361f911cc3f139f8a96dddadd0cfd377a3450644f1368bccbf83ea7 \
  --output "$RSNA_STATE_DIR/runs/report-labeling-llm-v3-dashboard-NEW"
python -m http.server 8799 --bind 127.0.0.1 \
  --directory "$RSNA_STATE_DIR/runs/report-labeling-llm-v3-dashboard-NEW"
```

Only serve the generated dashboard directory on loopback. Its HTML contains private report text and model responses; keep it out of Git. The separate `technical_summary.json` has aggregate counts only. `valid` means structural/evidence acceptance, not a correct clinical label. All semantic-review counts remain zero until the independent blinded workflow is actually completed. The dashboard refuses synthetic sessions and existing output directories.

Six synthetic-fixture tests exercise changed-file and extra-file rejection, exact raw/reparse consistency, missing-work denominators, no not-mentioned-to-negative conversion, synthetic-session rejection, escaped report content and overwrite refusal. They test software behavior; they are not model results.
