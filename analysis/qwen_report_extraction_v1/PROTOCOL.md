# Qwen report-extraction focus v1

Status: **LOCAL audit and prospective preparation only. No new inference or GPU adapter.** MedGemma development is paused by user direction on 2026-09-15. The MRI/YOLO stage remains paused.

## Objective and scope

First verify the two saved Qwen v1 repeats and identify lexical failure mechanisms without changing predictions. Then prepare the exact existing, never-run Qwen v2 candidate against the exact v1 control. This is a whole-instruction-package comparison: both the task prompt and the appended definition clarifications differ. It does not isolate a single wording effect or introduce a new clinical ontology. The [machine-readable diff](configs/prompt_diff.json) includes every textual change and substantive change categories.

Keep the exact 40-development/18-validation assignment and the same first five development reports. These five are already inspected and are engineering cases, not fresh evaluation. Read the development JSONL and split metadata; do not open validation reports or evaluate new organizer labels. Hash the source CSV as opaque bytes. The inherited analyst-review file already contains reference values; reading it does not constitute a new reference comparison or adjudication. All 189 historical tracked analysis files are checked against [protected_history.json](protected_history.json).

## Local audit

Verify source/split/development and both historical prediction hashes against the frozen v2 anchors. Verify original run completion/config/source manifests, five-study order, report hashes, case sidecars, and exact reparse of every response under the original v1 parser. Compare original repeats without accepting shared failures. Verify the 12-item inherited review queue against its saved hash, reports and predictions; keep every item unadjudicated.

Run a separately labeled parser-policy sensitivity analysis on the saved raw strings under frozen v2. Publish only status-transition counts; do not export replacement labels, recalculate organizer agreement, or modify original rows. Inspect unique source offsets for whitespace-only diagnostic matches. A recovered lexical match is not clinical evidence of correctness.

Reconstruct both prospective prompt packages using existing pinned tokenizer/config assets, offline only. Match all ten historical rendered control prompts and input counts. V1 did not save per-call input/output token IDs, so do not claim an original-ID roundtrip or independently recomputed EOS/completion check. Record newly tokenized inputs privately. No weights, generation, provider API, downloads or model adapter are needed.

## Exact prospective recipe

[Experiment config](configs/experiment.json): `Qwen/Qwen3-14B` at `40c069824f4251a91eefaf281ebe4c544efd3e18`, BF16, no adapter or quantization, batch one, greedy `do_sample=false`, one beam, seed 20260914, SDPA, `enable_thinking=false`, 8,192 input tokens, 2,048 new tokens, 300 seconds per generation. This deliberately preserves the historical research recipe, rather than adopting publisher sampling recommendations. Original [runtime dependency pins](../report_labeling_llm_v2/configs/runtime.json) remain required. CPU tokenization uses CPU PyTorch; future CUDA settings require separate verification.

Use the native Qwen template. No MedGemma thought-envelope removal, v3 evidence-first prompt, constrained decoding, extra examples, regeneration, retry, preferred repeat or fallback. The control prompt/definitions are byte-identical to v1; candidate prompt/definitions are byte-identical to existing v2. They occupy the same template role and report placement; length/content differences are part of the whole-package comparison, not a schema-placement ablation.

Proposed order: control-1, candidate-1, candidate-2, control-2, each with the same five reports: **20 planned generations / 240 cells, five unique studies**. A later bounded execution implementation must record raw outputs, token IDs, settings, prompt/source hashes, completion status, runtime and all failure receipts. Any incomplete generation stops the paired comparison; preserved completed calls are descriptive partial evidence. Completed schema/evidence failures remain in denominators and do not trigger repair. This package has no execute command and cannot launch that proposal.

## Shared acceptance and states

Both future arms use the exact frozen [v2 parser](../report_labeling_llm_v2/scripts/core.py) as primary technical contract; apply the frozen [v1 strict parser](../report_labeling_llm_v1/scripts/benchmark_core.py) to both as a sensitivity analysis. Both parsers' source hashes are recorded. Do not compare historical v1-under-v1-parser directly against prospective candidate-under-v2-parser as if that isolated model improvement.

Require all 12 condition keys and nested `label`, `evidence_text`, `confidence`. Four states: positive (explicit target-supported finding), negative (explicit absence or below-threshold finding), uncertain (relevant ambiguity), not_mentioned (no relevant statement). **Never convert not_mentioned to negative.** Technical failures carry null accepted labels and a separate status; they are not a fifth clinical state.

V2 permits only one complete outer Markdown fence and unique exact or narrowly ASCII-whitespace-equivalent evidence, with original source span/offsets retained. It rejects repeated/ambiguous spans, translated/paraphrased/fuzzy evidence and additional structure; not_mentioned requires empty evidence and zero confidence. V1 is stricter about whitespace/fences but allows first-match repeated evidence and does not enforce zero not_mentioned confidence. These changes can increase OR decrease acceptance. No parser diagnoses anatomy, polarity or severity. Confidence remains uncalibrated and is never an acceptance threshold.

## Prospective decision criteria

1. Account for all four runs and every planned/attempted/unrun cell. Report failure types, exact versus whitespace spans, raw repeat identity, valid repeat identity, output lengths and timing.
2. Retain the existing strict advancement gate: **60/60 technically valid and identical cells across repeats for a prompt**, no truncation/context/OOM failure, and no schema regression. ChatGPT suggested a looser 58/60 threshold; this proposal does not adopt it. A technical pass alone never permits scaling.
3. Review report/evidence entailment before showing organizer answers. A qualified reviewer must examine all unique technically valid outputs from both repeats (deduplicate only exact equivalents with mappings retained). Blind arm/repeat and initially hide organizer labels. Record anatomy, polarity, severity, timing and unsupported-negative errors. The existing 12 items remain unresolved until qualified adjudication.
4. Show binary coverage, semantic abstentions and review outcomes; more abstention alone is not improvement. Require no increase in serious semantic errors before further development. Organizer agreement is a later secondary, descriptive comparison; do not force report-derived labels to match questionable reference labels.
5. No prompt winner, significance, clinical reliability or multilingual generalization from five cases (four English, one Spanish). Exact-repeat stability under greedy decoding is not robustness. No Bulgarian or other new language is evaluated here.

Full-40 development, the reused 18-study exploratory validation, bulk extraction and image training require separate decisions. The 18-study set was historically inspected and cannot become fresh validation by preserving its split. Fresh independently adjudicated evidence is still needed before scaling. No paid-resource authorization is created by this local work.
