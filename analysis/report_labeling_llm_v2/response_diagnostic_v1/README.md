# MedGemma response diagnostic v1 — offline only

**Prefix removal alone is insufficient.** The first two saved answers become valid JSON after removing one exact leading thought envelope, but four of 24 entries still fail evidence checks. Technically accepted entries also contain polarity/specification problems. The third response exhausts the token cap during its answer. [Results and recommendation](RESULTS.md) · [aggregate measurements](aggregate.json).

This is a post-experiment diagnostic of the three saved RTX smoke responses. It never loads model weights, generates text, invokes a provider or changes official v2 predictions. The original smoke remains FAILED / INCOMPLETE with zero accepted labels. The 40-development / 18-validation split and both v1 directories remain unchanged.

## Method

1. Verify all 17 original remote artifacts against the pinned inventory via the existing partial diagnostic reader. Recheck executed-code/preparation hashes and exact saved raw/parsed correspondence.
2. Verify all ten local tokenizer/processor/config asset SHA-256 hashes against the earlier pinned receipt. Load only the offline processor/tokenizer for `google/medgemma-1.5-4b-it` at revision `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`.
3. Decode saved output token IDs, both with terminal EOS and without only the verified terminal EOS. Require byte-identical correspondence with both recorded text fields. Locate delimiter IDs, count thought/answer/delimiter/EOS tokens separately and retain original completion status.
4. Hypothetically remove exactly one leading, nonempty `<unused94>…<unused95>` envelope. Require one opening and one closing marker globally and a nonempty answer suffix. Reject non-leading, missing, repeated or nested markers. Preserve suffix bytes exactly; do not strip arbitrary prose, search for a JSON substring or repair output. Call the unchanged v2 parser with the original generation status; truncation remains rejected.
5. Inspect the pinned template AST and generation config. Render the five fixed inputs with no extra control and, solely as negative controls, with unsupported `enable_thinking=False` and `True`. Compare rendered text and input IDs. Recheck against actual saved GPU inputs for the three attempted cases. No claim that an ignored kwarg suppresses reasoning.
6. Keep hypothetical technical outcomes separate from official predictions and clinical correctness. Inspect limited examples for specification contradictions; do not change references or calculate a new diagnostic-accuracy score.

All code is outside the frozen executed v2 scripts/configs/prompts/tests/PROTOCOL folders. The old completed-session gate and partial viewer remain untouched.

## Reproduce locally

Use the existing CPU tokenizer environment with the versions in [aggregate.json](aggregate.json), or recreate those versions in a separate environment. The small pinned assets must already be restored in the explicit cache. Missing or changed assets fail; this script performs no downloads. No Hugging Face token is needed for local cached inspection.

```bash
python analysis/report_labeling_llm_v2/response_diagnostic_v1/inspect_responses.py \
  --prepared state/runs/report-labeling-llm-v2-ada-executed-20260915 \
  --cache state/cache/tokenizer-preflight \
  --output state/runs/report-labeling-medgemma-response-diagnostic-NEW
python -m http.server 8796 --bind 127.0.0.1 \
  --directory state/runs/report-labeling-medgemma-response-diagnostic-NEW
```

Open `http://127.0.0.1:8796/response_diagnostic.html`. The private visualization shows token-budget bars, original-language report, exact answer suffix, full raw response, organizer reference and official versus hypothetical technical outcomes. Do not publish rendered HTML or private details. Summary JSON contains no report text or study IDs. Output directories cannot be overwritten; use `RSNA_STATE_DIR` for relocated private state.

The complete repository test command now includes the separate diagnostic guard tests:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

The 17 new synthetic checks exercise suffix-byte preservation, malformed/repeated/nested envelopes, mid-sequence EOS, token accounting and refusal to rescue truncated generations. They do not measure model quality.
