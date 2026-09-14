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
