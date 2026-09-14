# Design review record

On 2026-09-14 the user asked Codex to report progress to their ChatGPT browser conversation and obtain direction. Codex sent a public-safe description of the proposed implementation, dataset counts and model metadata. No raw reports, study identifiers, organizer labels at study level, private predictions or credentials were sent.

ChatGPT reviewed the design and advised proceeding with implementation while preserving NOT RUN status and waiting for resource approval before inference. Its feedback is advisory; it does not authorize spending, publication of private data or changes to the user constraints.

Implemented or documented in response:

- Treat the reused 18-study partition as exploratory; no scaling certification from it.
- State possible pretraining contamination and unequal model capacities.
- Keep join IDs and reference labels out of model prompts.
- Add all-58 tokenizer/context preflight and retain official rendered templates, revisions and stop-token metadata.
- Require evidence for both positive and negative statements; silence is not negative.
- Keep infrastructure and schema/evidence failures separate from medical abstention.
- Preserve the primary first pass; describe a generated format repair as a secondary inference pass that can change labels.
- Require a repeatability check on five development studies for both models before freezing.
- Use correct yield over all checks as the primary endpoint, with coverage and conditional error.
- Evaluate binary agreement as an empirical selection rule with potentially correlated errors.
- Keep language denominators explicit and preserve the existing analyst language review.

One suggestion was not adopted for primary acceptance: broad Unicode/whitespace/punctuation normalization of quotations. The user requested verbatim evidence, so v1 uses exact source spans, records evidence failures and documents possible false rejections. A future normalization sensitivity analysis must be separately specified; it must not silently change accepted labels.

Model/runtime code remains untested against actual weights and CUDA. The review is not a claim that the models work, that clinical extraction quality is established or that the benchmark has completed inference.

## Review of the implementation commit

ChatGPT reviewed public commit `0ec86743265dc9b9355985844835a1ddb306dadf` after GitHub CPU CI passed. It identified two changes before the development smoke test, both adopted locally and covered by synthetic regression tests:

- `verify_preflight()` now rejects any model with an overflowing input; the tokenizer-preflight command also exits unsuccessfully after saving its diagnostic manifest. Runtime overflow handling remains for later checks and longer second-pass prompts.
- MedGemma now uses the processor's `apply_chat_template(..., tokenize=True, return_dict=True, return_tensors="pt")` path for input construction, preserving processor-specific tensors. Both preflight and inference call that same path. Qwen retains its non-thinking template and no-truncation tokenizer path. This follows the [publisher's usage example](https://huggingface.co/google/medgemma-1.5-4b-it); actual CUDA compatibility still awaits the approved smoke test.

The review recommended retaining the prompts, checkpoints, decoding recipe, statistical design and exact split for the smoke test. It also flagged two follow-ups before validation: explicitly review development technical-failure rates before accepting the execution freeze, and distinguish duplicate nested fields from duplicate condition keys in failure reporting. Those are recorded as pending; the current automatic freeze checks do not impose a maximum development failure rate, and nested duplicate keys currently share the `duplicate_condition` category. Do not mistake the passing synthetic checks for a completed execution freeze or authorization to run validation.

## Completed smoke review and offline follow-up

The earlier paragraphs describe pre-execution reviews. The bounded stage subsequently ran at commit `f334958` on the approved A100 and is now complete; [RESULTS.md](RESULTS.md) is the current status.

Codex sent aggregate smoke counts, measured runtime/memory, repeatability failures, the matched five-study reference comparison and stop confirmation to the same user-authorized ChatGPT conversation. No source reports, study IDs, raw predictions or credentials were sent. ChatGPT agreed that neither the full 40-study run nor bulk labeling should proceed, and recommended inspecting saved responses locally before considering a new version. It did not independently execute or clinically adjudicate the experiment.

The offline follow-up inspected all five MedGemma first responses and all nine Qwen binary mismatches, two evidence errors and one uncertain output. A detail in the advisory response was corrected against saved evidence: **two** MedGemma responses were outer-fence-only, not three. Only one of those would pass every existing schema/evidence check after a diagnostic fence removal, and semantic contradictions remain. One truncated response has an earlier complete object followed by incomplete repetition; the other does not complete an object. No diagnostic response became an accepted prediction.

The resulting recommendation is a prospective, separately versioned development candidate with narrow structured-output handling and anatomy/severity/evidence safeguards. No prompt tuning, parser change, additional model call or validation evaluation was performed after seeing the smoke. The candidate should first pass local synthetic checks and a bounded, approved five-case gate; full development remains blocked on technical and semantic review. A new independent adjudicated set is still needed for scaling claims.

ChatGPT's suggested engineering gate—technically usable 60/60 cells under a predeclared policy, no truncation/context/OOM, exact repeatability, no unexplained evidence errors and semantic review—is documented as a proposal, not an executed v2 or a clinical validation threshold. Private diagnostic tables and a v2 proposal are included in the local review package.

A final aggregate follow-up corrected the fence count and reported the offline findings and rule/Qwen intersection. ChatGPT accepted the interpretation with wording cautions: keep the disagreement categories tentative, do not infer organizer errors, distinguish quotation-contract failures from clinical mistakes, and do not claim agreement is generally worse from five studies. These qualifications are retained in RESULTS. No unsent prompt remained after the final review.
