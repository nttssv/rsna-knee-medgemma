# Prospective v3 protocol — local candidate, NOT RUN

## Question and experimental unit

Does an evidence-first prompt reduce MedGemma polarity, anatomy and severity contradictions while maintaining technically valid extraction yield? This is a paired engineering comparison on **five already-inspected development studies**. It cannot establish generalization or clinical superiority. Twelve condition cells and repeated outputs from one report are correlated, not independent studies.

The control is **v3-control: exact v2 prompt plus shared v3 engineering controls**. It is not the historical v2 execution. The candidate adds an explicit evidence-first decision ladder and brief synthetic contrasts. Both receive the exact same unchanged target definitions, five original-language reports and JSON schema. Neither receives MRI images, translations, study identifiers, organizer answers or previous model predictions in its prompt.

## Arms and common controls

Both arms use pinned `google/medgemma-1.5-4b-it` revision `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`, BF16, SDPA, batch one, greedy decoding, one beam and seed 20260914. Keep the ten runtime versions from the executed v2 recipe. No quantization, adapter training, model substitution or Qwen run is part of this candidate.

The planned input cap remains 8,192 tokens; the shared output cap becomes **4,096** with a 300-second generation limit. The cap is a newly chosen development hyperparameter, not demonstrated sufficient. It is applied equally to both arms and does not alter old receipts. No working MedGemma thinking-disable flag is claimed.

Planned whole-run order is **control-1 → candidate-1 → candidate-2 → control-2**, retaining original report order within every run. This balances early/late positions by arm but is not randomized allocation or proof against all temporal effects. Use the same GPU/driver/runtime and separate model processes for each run; log hardware and timings. Greedy repeats check reproducibility, not independent predictive variance.

Maximum design: 5 reports × 2 arms × 2 repeats = **20 primary generations**. Framing and budget are held constant across arms, so this comparison cannot separately estimate their effects. Historical A100, RTX, v1 and v2 outcomes remain descriptive background only.

## Output and shared acceptance

Every report returns one object with the same 12 keys and exactly `label`, `evidence_text`, `confidence` per key. `positive`, `negative`, `uncertain` and `not_mentioned` retain their original meaning. `not_mentioned` requires empty evidence and confidence zero; never convert it to negative. Confidence remains uncalibrated.

For completed generations only, optionally remove exactly one nonempty **leading** `<unused94>…<unused95>` envelope. Require one opening and one closing marker in the entire response and a nonempty suffix; reject nested, repeated, malformed or mid-response markers. Preserve the suffix bytes and exact original raw output. Responses without markers pass directly to the frozen v2 validator. Noncompleted generation is rejected before any envelope handling.

After framing, reuse the unchanged v2 whole-JSON/outer-fence, duplicate-key, schema and unique evidence-span logic, including its narrow whitespace fallback and original report offsets. Do not extract arbitrary JSON substrings, retry, generate a repair or select a preferred repeat. The parser remains lexical/structural: it can accept a wrong label attached to an exact quote. Never add an automatic medical label-flipping rule.

## Semantic review and measurements, predeclared

After a future complete matched session, a qualified reviewer must inspect the full original report, target definitions, each candidate label and evidence. Assess model-versus-report entailment before showing organizer answers where feasible, and conceal arm identity during initial review. Record reviewer identity, time, output hash, category and notes. The prepared 240-row template is blank; it is not completed review.

Count these potentially overlapping categories: polarity contradiction, wrong anatomy/compartment, severity-threshold error, temporality error, unsupported negative, other entailment failure, reference discordance and insufficient information. `no_issue_identified` is exclusive of error categories. These are human-review categories, not automatically inferred truth. Organizer/report discordance is not itself evidence that the organizer is wrong.

Report separately by arm/repeat and condition:

- Generation completion, JSON acceptance, evidence acceptance and technical failure counts, with attempted and planned denominators.
- Four-state distribution, binary coverage, unresolved/abstaining cells and all reviewed semantic error categories.
- Reviewer-confirmed, technically accepted label yield relative to all planned cells; unreviewed cells never count as correct or pass a semantic gate.
- Organizer agreement conditional on a binary decision, clearly distinguished from report entailment and coverage; preserve disagreements without changing references.
- Repeated-output consistency, including state/evidence/offset differences, without selecting a better repeat.
- Tokens, seconds and GPU memory, including thought-envelope overhead where present.

Show per-study descriptive outcomes; no significance test, superiority claim or generalization estimate from n=5. Language breakdown is descriptive (four English, one Spanish), not multilingual validation. Technical and semantic performance must both be visible; a prompt that abstains on everything cannot claim success from fewer errors.

## Gates and stopping

This version contains **no GPU execution adapter** and always reports NOT_EXECUTION_READY. Preparation, tests, tokenizer checks and ChatGPT advice do not unlock inference. Before any later paid run, implement and review a separate bounded adapter with exact candidate/plan/cache checks, raw-first receipts, parent/worker provenance, process deadlines and externally verified provider shutdown. Obtain a current quote and explicit resource authorization for that reviewed version; the earlier smoke's approval is not reused automatically.

Future noncompleted generation should stop the session and preserve partial evidence. An incomplete session cannot produce the paired comparison. Completed technical failures remain visible, not repaired. Expansion stays blocked unless both arms/repeats are fully accounted for, repeat consistency is assessed, qualified review is complete and unresolved critical semantic errors are addressed in a separately reviewed decision. Nothing here automatically permits full-40 development, 18-study validation, bulk labels, MRI training or YOLO distillation.

## Preservation and limitations

All 124 historical tracked analysis files are anchored at commit `93259b6256589f38478d1375a10394157ea23d8b`. The original 40/18 split is checked by hash and membership; preparation opens only membership plus the fixed five-report development package. No validation reports or labels are evaluated. The 18-study holdout has previously been inspected for the frozen rule baseline, so it is not a universally untouched benchmark. Never tune this candidate on its labels; any future evaluation needs a separately reviewed plan and must acknowledge small per-condition support.

The 24 synthetic cases are assistant-authored expectations, including a few language contrasts, and have not received qualified clinical review. Test success proves serialization/guard behavior only, not that MedGemma follows the prompt. Identical target-definition text establishes textual preservation, not independent clinical validation. All 12 inherited development adjudication items remain unresolved.
