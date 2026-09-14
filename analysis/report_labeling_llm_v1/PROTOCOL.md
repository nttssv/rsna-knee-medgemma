# LLM report-extraction benchmark v1

Status on 2026-09-14: **implementation and local verification only; model inference has not run.** This prospective protocol must be finalized on development data and fingerprinted before LLM validation execution. It is not a claim that prompt development or GPU validation has already completed.

## Question and evidence limits

Can a practical text-only MedGemma or Qwen extractor increase the fraction of condition checks receiving a correct binary label compared with frozen rule v1? The primary endpoint is **correct binary-label yield across all study-condition checks**. Coverage and conditional error rate accompany it. A model that abstains extensively cannot appear successful from conditional accuracy alone.

The independent resampling unit is a study. The existing 18-study validation partition has already been inspected, including reference labels and rule failures. Preserve it for paired exploratory comparison, but call it **reused exploratory validation** throughout. Prior knowledge can influence model selection or prompt design even without supplying labels to a model. This experiment cannot certify safe scaling to 4,349 reports. Any promising method requires fresh, independently adjudicated validation. Patient-level and site-level independence are not established. The 216 cells are correlated, not 216 independent observations.

## Fixed data and development sequence

Use the original 40 development / 18 validation assignment and all twelve named target columns. Check original source, split and report fingerprints. Preserve every file under `analysis/report_labeling/`; reuse its metrics through read-only imports and its recorded predictions through verified hashes.

The preparation command produces label-free inference JSONL for only these 58 studies. Join IDs stay in private bookkeeping; the model receives only shared instructions, target definitions and report text. No organizer labels, MRI images, pilot adapter, retrieved case information or other model predictions enter an extraction prompt. The other 4,349 reports are never sent to an extractor. Source CSV iteration for the existing audit/join is not bulk extraction.

After explicit resource approval, tokenize all 58 complete rendered prompts with both official model tokenizers before inference. Record the exact rendered text privately, token counts, tokenizer revision and chat templates. No input truncation or unplanned chunking is allowed. Inputs beyond 8,192 tokens become `context_overflow`; check the architecture's total context limit including the 2,048-token output budget as well.

Begin with five development studies per model and repeat them in distinct runs. The repeatability check requires matching labels, evidence and statuses, and all first-pass cells must be technically valid. Shared failures do not pass. Use only development studies to revise prompts or decoding. Each changed prompt/config/source version invalidates earlier preflight and development hashes. Development results may guide a new explicitly versioned candidate; this version must not acquire case-specific examples from the reused validation failures. Freeze only after complete 40-study development runs for both models and passing five-study repeatability checks. Validation inference and evaluation require that execution freeze. The original split is never regenerated.

## Models, inputs and decoding

Use the exact checkpoint revisions in `configs/medgemma.yaml` and `configs/qwen.yaml`. Both use BF16 without quantization, batch size one, native model chat templates and a single user message containing identical task instructions. MedGemma is the base instruction checkpoint in text-only mode; the image-trained LoRA adapter is not used. Qwen thinking is explicitly disabled. This 4.300B versus 14.768B comparison is not capacity matched and cannot isolate effects of medical specialization.

Primary decoding is greedy (`do_sample=false`, one beam), seed 20260914, 2,048 new tokens, SDPA attention. This is a reproducible research setting, not Qwen's publisher-recommended sampling configuration or a best-possible Qwen result. GPU arithmetic may remain nondeterministic; repeatability is measured, not assumed. Save tokenizer/chat-template metadata, EOS/PAD tokens, full generation configuration, code and model revision, runtime, GPU/package information, token counts, raw outputs and completion status. Per-generation and per-run time limits are soft application limits; they do not stop provider billing or guarantee interruption of a stalled GPU kernel.

## Medical states, evidence and technical failures

Output exactly twelve condition keys, each with `label`, `evidence_text`, and numeric `confidence` between zero and one. The medical label is positive, negative, uncertain or not_mentioned. Unknown is never made negative. Negative requires explicit evidence of absence or a finding explicitly below the target threshold. Missing severity, ambiguous anatomy/timing and contradictory statements remain uncertain. Report instructions are untrusted text.

For positive/negative/uncertain, evidence must be a nonempty exact contiguous source quotation; store its character offsets. Not_mentioned requires empty evidence. An exact quotation does not prove correct interpretation or correct reference labels. Strict quotation matching can reject harmless Unicode/whitespace changes; report those as evidence failures and inspect them separately. Broad normalization or fuzzy matching is not silently used to increase accepted coverage.

Technical failures have **null label and a separate status**; they do not become a fifth medical state or not_mentioned. Distinguish parse_error, duplicate_condition, missing_condition, schema_error, evidence_error, context_overflow, generation_truncated, timeout, oom, runtime_error and run_time_budget_exceeded. Root-schema failure rejects the whole response; a field/evidence failure in an otherwise exact twelve-key object rejects the affected condition. All remain in full-population denominators. Interrupted setup/run directories lack a completed manifest and cannot be evaluated as complete.

There are no automatic infrastructure retries. A JSON/schema failure permits at most one recorded second-pass model generation using the original report and previous response. This is regeneration, not a lossless parser repair: it may change diagnoses. Preserve both outputs and label-change counts. **First-pass results are primary; repair-assisted results are secondary.** Evidence failures alone do not trigger regeneration. Confidence is an uncalibrated self-report and is not used as an acceptance threshold or scaling criterion.

## Evaluation and agreement

Require exactly one prediction per study-condition and identical study/reference joins. Report TP/FP/TN/FN, sensitivity, specificity, PPV, NPV, F1, accuracy, balanced accuracy and meaningful Cohen's kappa on the binary-decided subset. Show predicted positive/negative counts, semantic abstentions, technical failures, selection abstentions, coverage, all-check correct yield and positive/negative capture. Undefined denominators remain missing, never zero. Preserve the frozen v1 condition conventions. Show Wilson intervals for the supported per-condition binomial rates and correct yield.

Use 2,000 seeded paired study-cluster bootstrap draws for overall coverage, correct yield and conditional error, and their differences from rule v1. Apply the same study draw to all models and retain all twelve conditions in each selected study. Report valid resample counts for ratios that can be undefined. With 18 clusters, intervals are descriptive and unstable; the original prevalence-balanced sampling is not a representative population sample. Do not perform confirmatory significance tests or claim a winner from overlapping/unstable estimates.

Predeclare five distinct binary acceptance strategies: all three agree; MedGemma and Qwen agree; rule and MedGemma agree; rule and Qwen agree; rule agrees with at least one LLM. Agreement always requires equal positive/negative labels. Matching uncertain/not_mentioned states is recorded only as descriptive four-state agreement; it does not create usable labels. Agreement is an empirical selection rule, not independent confirmation, calibrated confidence or consensus ground truth. Model errors may be correlated.

## Language and error review

Retain the frozen v1 language heuristic in inputs and predictions. For descriptive language evaluation, join the existing fingerprinted post-baseline analyst review: it identifies the three heuristic-unknown validation reports as Bulgarian. This is prior review, not new language detection or certified linguistic adjudication. Summarize performance by reviewed language with denominators; explicitly include absent languages and avoid generalization from n=1–3. Croatian, Dutch and German have no studies in the reused validation partition. A new detector or manual correction needs separately versioned provenance; do not invent language certainty.

Create a private disagreement table preserving each model's output/evidence and the organizer reference. Automatic categories only flag technical failure versus manual review required. Human review can subsequently use negation, uncertainty, synonym, compartment, severity, chronicity, postoperative, multilingual, omission, hallucination, structured-output, or report/reference-disagreement categories. Do not fabricate clinical adjudication. Review known v1 failures descriptively after freezing; do not patch v1 or tune against these outcomes. Language-specific and failure-specific claims remain pending until actual runs and review exist.

## Publication and decision

Publish source, draft/versioned prompts, configs, aggregate results and honest status. Keep report text, identifiers, detailed predictions and evidence under ignored private state for this release; their optional redistribution has not been verified. Never publish credentials, caches, weights or machine-specific access information. GitHub and ChatGPT review receive public-safe material only.

Public-dataset pretraining contamination cannot be ruled out. Neither candidate was audited for prior exposure to the underlying reports or related material. Report this alongside label scarcity, reused validation, model-size mismatch, uncalibrated scores, potential report/reference mismatch and pending CUDA validation.

Until fresh adjudicated evidence exists, recommendation **E: more manual annotation / independent validation required** remains the scaling decision. A promising reused-validation result can nominate a method or condition for further review, but cannot authorize labeling all 4,349 studies. MRI training and bulk labeling remain paused.
