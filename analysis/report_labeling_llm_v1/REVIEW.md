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
