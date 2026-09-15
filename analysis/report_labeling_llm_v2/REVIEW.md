# V2 review and unresolved questions

The user authorized the next local step after reading the detailed instruction in their ChatGPT browser conversation. That instruction requested a separately versioned implementation, immutable v1 evidence, deterministic outer-fence handling, narrow evidence normalization, generalized semantic prompting, synthetic tests and a private adjudication queue, followed by a bounded proposal and a stop before GPU execution.

Implemented design decisions:

- Normalize only four explicitly named ASCII whitespace characters. Accept a unique exact occurrence first; use unique normalized matching only if no literal occurrence exists. Preserve exact original source spans and Unicode code-point offsets.
- Reject noncompleted generation before any syntactic rescue, including a complete earlier object inside a truncated response. No substring extraction or generated repair is part of the primary policy.
- Apply the same output policy to both models; retain paired common prompts. No teacher-specific exception.
- Keep the semantic review cases as reviewer expectations. The parser does not pretend to diagnose or enforce medical entailment; tests explicitly demonstrate that lexical validity can coexist with an incorrect label.
- Keep all 12 inherited development review candidates unresolved. No qualified adjudicator was available in this task, and no organizer label changed.

Questions recorded at the first candidate publication (see the adapter follow-up below for implementation status):

1. Verify the MedGemma official template's supported response controls and EOS behavior. The v1 flag named enable_thinking was not passed to its MedGemma template. V2 makes no claim of disabled reasoning and does not silently change the decoding recipe.
2. Implement a bounded inference adapter that actually uses the v2 contract, logs rendered prompts and revisions, and permits only the exact five studies. This local package deliberately has no model execution entry point; a model/config file alone is not a working CUDA integration.
3. Fresh tokenizer preflight must cover only the selected five development inputs in this stage. Longer prompts and optional runtime controls require new fingerprints; historical v1 preflight cannot certify them.
4. The unique-span policy may reject legitimate repeated quotations. Keep failures visible; review whether the model can choose longer contiguous evidence before considering another prospectively documented policy.
5. A report can contain relevant contradictions outside one chosen passage. Full-report semantic review is required. Whitespace normalization addresses lexical formatting only and cannot resolve report/reference or target-definition ambiguity.
6. A qualified human must review unresolved development references separately from software testing. Do not interpret synthetic fixtures or ChatGPT advice as clinical adjudication.

The experimental-design guidance is used to preserve study-level denominators and label this as a repeated engineering comparison. No new split, randomization or statistical superiority claim is introduced. Advisory browser feedback cannot authorize compute.

## Browser review of the implemented candidate

ChatGPT reviewed the aggregate local implementation description. It recommended exact-first evidence matching, a human semantic-review artifact with identity/time/output provenance, and testing JavaScript highlighting against Python code-point offsets. The first two suggestions were adopted prospectively before any v2 generation. The viewer already uses `Array.from(report)` for code-point slicing; a non-BMP synthetic check is included in verification. It agreed that an absent GPU adapter does not prevent publishing a NOT RUN local candidate, but adapter implementation, fake-backend tests, dry run and separate review remain prerequisites before requesting paid compute. No raw reports or identifiers were shared in this review.

A second advisory review found no substantive blocker to NOT RUN publication after repository checks. It requested binding human review to the complete experiment fingerprints in addition to each row, and making repeat mismatch block advancement explicitly. Both are implemented and tested; no human review is claimed. Browser verification confirmed all 12 rows/48 pending model cells and exact highlighting after a synthetic non-BMP character. Advisory review was of the implementation description, not an independent code execution or clinical assessment.

## Local adapter follow-up

The user authorized the next local implementation step after publication. The adapter, fixed-five dry plan, offline-only loaders, durable raw output recording, process deadlines and fake-backend tests are now implemented; see [RUNTIME.md](RUNTIME.md). The original prompt/ontology/acceptance code remains unchanged. Real tokenizer/model execution is NOT RUN and GPU flags remain false. MedGemma PAD 0 is now preserved using a None check; the v1 PAD fallback is documented as a runtime difference.

ChatGPT reviewed the adapter design description and requested explicit offline environment/cache isolation, an exact parser-decoding rule, process-tree timeout tests and full plan-hash coverage. These were implemented: missing cache snapshots fail before Transformers import; only a terminal verified EOS ID is removed before decoding; workers own process groups; actual CPU watchdog tests cover a descendant; all executable scripts, runtime/package/token limits, model configs and prompt/ontology files participate in candidate/plan fingerprints. Its review distinguishes this local publication milestone from unperformed real-cache and CUDA checks.

The subsequent published-source review identified two future-unlock blockers: standalone worker access and missing plan/session binding in child outputs and the viewer. Workers now require an inherited parent pipe/nonce, parent identity and ordered dispatch; each child and parent result records the reviewed plan/session hashes, and the viewer requires the complete matching session. The dry plan’s non-enforcing `compute_authorized` field was removed. No execution flag was enabled. Tests cover both fixes; cache-content verification beyond template/EOS/PAD remains explicitly unclaimed.
