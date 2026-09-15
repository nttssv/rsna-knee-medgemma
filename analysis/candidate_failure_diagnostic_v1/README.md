# Candidate failure diagnostic v1

A read-only, CPU-only forensic audit of the failed MedGemma v3 execution. **All nine input/output token reconstructions passed.** The short candidate responses omitted evidence/confidence objects; the long response repeated a wrongly shaped array until truncation. See [RESULTS.md](RESULTS.md) and [PROTOCOL.md](PROTOCOL.md).

Source model: `google/medgemma-1.5-4b-it`, revision `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`. No model/backend is instantiated. The already available processor/tokenizer assets and original [ten dependency pins](../report_labeling_llm_v3_execution_v1/configs/requirements.txt) are used, with CPU PyTorch 2.8.0 for the local audit. The original inference used 2.8.0+cu128. Asset hashes and actual local versions are in [summary.json](aggregate/summary.json).

Restore the private v3 execution package and operator inventory from the existing private backup. Point `RSNA_STATE_DIR` to that state and use an environment containing the pinned packages:

```bash
python analysis/candidate_failure_diagnostic_v1/audit.py \
  --prepared "$RSNA_STATE_DIR/runs/report-labeling-llm-v3-execution-executed-20260915" \
  --cache "$RSNA_STATE_DIR/cache/tokenizer-preflight" \
  --inventory "$RSNA_STATE_DIR/runs/report-labeling-llm-v3-private-provenance-20260915/remote-file-hashes.json" \
  --output "$RSNA_STATE_DIR/runs/candidate-failure-diagnostic-NEW"
```

The cache must already contain the exact pinned tokenizer/processor assets. The script never downloads assets or weights, accesses a provider, reads organizer label values or runs generation. `summary.json` is aggregate; `details.json` remains private. The original 28-file inventory, executed sources, prepared inputs and frozen predictions are verified and preserved.

Twelve synthetic counterexample tests cover exact EOS handling, corrupted input/render/output rejection, completion-status mismatch, separation of thought and answer tokens, repeated answer text, refusal to rescue string-state outputs, child receipt checks, extra dispatch rejection and rebound wrong attempt counts:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_candidate_failure_audit.py
```

The saved-receipt audit does not recreate the past live process/pipe or rehash remote model weights. It checks the original cache receipt, recorded metadata and their transfer bindings. Its findings do not establish medical correctness, causal effects of prompt wording or permission to scale.

All 462 CPU tests passed (450 prior plus 12 new). The two-file private diagnostic archive was restored and checksum-verified. [Verification record](aggregate/verification.json). Historical experiment outputs, the 40/18 split and all 181 protected analysis files remain unchanged.
