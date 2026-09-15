# Five-report generation candidate — prepared locally, NOT RUN

This candidate follows the successful [load-only preflight](LOAD_PREFLIGHT.md), executed at `91bdbbd`, with results published at `d769e7e`. The user authorized local preparation of the next prediction experiment. No additional GPU start or inference is authorized by this document or by external review.

## Objective, input and output

Test whether each model can return technically valid, repeatable, evidence-grounded extractions under the prospective v2 contract. This is a five-study development feasibility check, not validation accuracy or MRI training.

**Input:** the same five original development reports (four English, one Spanish), in their original order and language. Each model receives the common instructions, 12 target definitions and one JSON-quoted report. No images, translations, study IDs or organizer answers enter the prompt. Organizer labels remain separate, solely for private comparisons.

**Output:** one JSON object per report with 12 entries. Each entry contains `label`, `evidence_text` and uncalibrated `confidence`. Labels are `positive`, `negative`, `uncertain` or `not_mentioned`. Missing mentions never become negative. Record exact raw responses and generated token IDs before parsing, then preserve technical status and source evidence spans separately from label/reference agreement.

Exact order: **MedGemma-1 → MedGemma-2 → Qwen-1 → Qwen-2**, five reports per run, at most **20 primary generations** and 240 correlated condition cells. There are only five independent studies. Two repeats do not double the sample size. Each run loads in a separate process. No generated repairs, automatic retries, preferred-repeat selection, validation inference or bulk labeling.

## Narrow change from the completed load stage

Enable the existing four execution flags in the runtime, output policy and two model configs. These flags make the explicit generation command callable; they are **not evidence of paid authorization or successful inference**. Correct the stale MedGemma template-verification description to reflect the authenticated pinned preflight. `real_runtime_verified` and model/output status remain false/NOT_RUN because generation is untested.

Update tests that asserted permanently disabled flags to verify explicit-command refusal and each independent disabled gate instead. Retain subprocess watchdog, parent/child provenance, fixed-five inputs, no-overwrite, raw-before-parser, fail-fast and measured-viewer checks. No generation algorithm, parser, evidence rule, prompt, ontology, model revision, decoding setting or acceptance gate changes. Both v1 folders and the original 40-development/18-validation split remain unchanged.

The four flags and test changes alter candidate fingerprints. A fresh private preparation and execution plan are therefore mandatory. Historical packages and load receipts remain unchanged and bound to their original code; do not reuse their plans with this candidate.

## Exact recipe and reviewable preparation

Pinned IDs/revisions, template hashes, stop-token IDs and all ten environment versions are in [runtime.json](configs/runtime.json), [MedGemma](configs/medgemma.json) and [Qwen](configs/qwen.json). Keep BF16, batch one, SDPA, greedy decoding, one beam, seed 20260914, 8,192 input-token limit, 2,048 new-token cap and 300-second generation limit. Qwen uses `enable_thinking=False`; no equivalent MedGemma switch is claimed. [RUNTIME.md](RUNTIME.md) documents EOS handling, the corrected PAD 0 behavior and strict decoding.

```bash
python analysis/report_labeling_llm_v2/scripts/prepare_local.py \
  --output state/runs/report-labeling-llm-v2-generation-20260915
python analysis/report_labeling_llm_v2/scripts/run_smoke.py plan \
  --prepared state/runs/report-labeling-llm-v2-generation-20260915 \
  --plan state/runs/report-labeling-llm-v2-generation-20260915/execution_plan.json
```

These commands prepare a dry plan only. The fresh plan digest and exact published candidate must be reviewed before paid execution. The plan remains private because it contains study identifiers. The explicit future `run --execute` command additionally requires this exact plan SHA, unchanged fingerprints, an explicit offline cache and active parent/worker attestation. No command in this document starts a provider or generates predictions.

## Measured foundation and proposed resource bound

Both pinned caches passed all 13 required-file checksums per model. Both loaded on the existing A100 80GB in BF16, using 8.010 GiB / 27.508 GiB peak allocated memory. The entire load-only check took 140.61 seconds. These measurements do not establish KV-cache memory, generation speed, SDPA kernel execution or prediction quality.

On 2026-09-15 during this preparation, the authenticated RunPod console showed the existing A100 stopped at $0.00/hour and **Start for $1.59/hour**. Availability can change; do not silently create or migrate a pod if the existing one cannot restart.

Proposal: **30–45 minutes expected, at most 60 minutes from provider start, with a $1.60 compute ceiling**, excluding existing storage. The range uses historical v1 generation rates and is not a v2 measurement. Refresh the price immediately before any separately authorized start; if the quote would exceed the ceiling, do not start. Reuse the saved environment/cache; missing or mismatched assets cause a stop, not downloads or repairs.

Before starting, record the UTC start request and a provider deadline 60 minutes later. Use the established outer deadline wrapper to interrupt the parent by minute 55, leaving five minutes for partial-result transfer and provider shutdown. The parent retains its 1,800-second per-process and 3,600-second session bounds; the earlier outer deadline wins. The supervisor kills/reaps its worker process group on interruption. Stop the pod immediately on completion or runtime failure after prompt result transfer, and no later than the provider deadline even if transfer is incomplete. Do not let transfer or semantic review extend billing. The operator must verify the stopped UI; process timeouts alone cannot stop provider billing.

## Result review and next decision

After execution, preserve complete or partial private receipts, copy and hash-verify results, stop compute, then analyze locally. A complete four-run session may populate the existing private [case viewer](smoke_review_tools/README.md) with original report → raw response → parsed states/evidence → organizer/reference comparison. Show technical failures and abstentions explicitly. Incomplete sessions remain incomplete and cannot masquerade as measured four-run results.

Report per-model validity, four-state distribution, binary coverage and conditional reference agreement, evidence matching, tokens/time/memory, repeat consistency, per-condition results and descriptive language breakdown. Contrast with v1 and the rule baseline on the same five cases. These are development observations; neither reference agreement nor ChatGPT review establishes clinical correctness. All 12 inherited adjudication items remain unresolved. Qualified human semantic review is required by the unchanged [protocol](PROTOCOL.md) before considering expansion. No full 40-study, 18-validation-study or 4,349-report run is automatic.

## Local verification

All **231 tests passed**, including independent refusal by each of the four configuration gates and explicit-command refusal. All **70 frozen v1 files** and **478 pre-existing non-cache private files** verified unchanged. The five inputs, both prompt files, private references and unresolved adjudication queue match the completed load-stage preparation byte-for-byte. Production generation, parser and parent-session code are unchanged.

Fresh private plan SHA-256: `8d2926fd2d5ed8ee2b8a1a7ae37608f6149b2b4f24612d4013e531170f816b8e`. It contains exactly the prescribed four-run order and a 20-generation ceiling, with `DRY_PLAN_ONLY` and zero model calls.
