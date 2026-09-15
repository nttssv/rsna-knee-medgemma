# Qwen runtime — local results

**Runtime/review integration completed locally. No new Qwen output, MRI training, GPU use or paid compute.** The real five-report package is prepared but unexecuted. The completed demonstration uses fabricated reports and fabricated all-not-mentioned outputs; it is not a Qwen result.

| Verified local item | Result |
|---|---|
| Historical analysis preservation | 203 files unchanged, including the Qwen audit/design |
| Real inputs and prompts | Exact five development inputs, v1 control/v2 candidate, fixed40/18 split |
| Real execution | Blocked before any session/weight load; no permission flags unlock it |
| Synthetic parent/worker workflow | Four supervised workers, five fabricated outputs each |
| Durable demo ledgers | 20 attempt-start intents, 20 raw responses, 20 primary/secondary parsed responses |
| Technical demo gate | Fabricated60/60 valid-identical cells per arm; not model performance |
| Review workflow | 20 randomly coded outputs, 240 blank cells, organizer answers excluded |
| Semantic results | None; inherited12 Qwen review items still unresolved |

The private prepared package contains only label-free input/prompt/tokenization files, split membership and an audit receipt. It does not copy the historical details or review CSV that contains organizer answers. The blinded bundle similarly contains no arm/repeat fields, study identifiers or organizer labels. Report text itself may disclose context, and identical outputs or reviewer familiarity may weaken practical blinding; interface hiding is not independent validation.

Both parser responses are bound to the same raw record. Qwen EOS IDs [151645,151643] and the2048-token cap replace the old MedGemma assumptions in the reused runtime. The strict advancement gate includes confidence, source span/offsets and normalization metadata, so a change in any required field prevents valid-repeat identity. Synthetic counterexamples confirm that changed secondary results or missing attempt ledgers are rejected.

Attempt-start records precede backend calls. A crash before output therefore remains visible as a dispatched intent with no raw response; a parser exception preserves raw output. Incomplete generations stop subsequent dispatches. Parent timeout/interruption uses the existing tested process-group kill/reap helper. These checks cover software behavior, not remote GPU failures or provider billing. The partial inspector reports counts and malformed trailing records without promoting partial data to accepted predictions.

The review collection validates explicit semantic categories for technically valid cells and an unreviewable disposition for technical failures. It never treats blank fields as success. Finalization exports a blank organizer-comparison template with all reference fields null and status DEFERRED_NO_REFERENCE_LABELS_LOADED. Qualified reviewer credentials and later independent clinical evaluation remain outside this software's guarantees.

The full CPU suite passed **546 tests**, including62 new runtime/review checks. These use fabricated inputs; they do not demonstrate that Qwen will satisfy the protocol. Browser inspection confirmed the synthetic banner, two-pane input/output layout,20 output choices,12 conditions and working condition selection.

## Next decision

The next separate milestone is a concrete Qwen GPU execution proposal with an approved resource cap, pinned full-cache verification, tested provider shutdown controls and external stop verification. The real adapter has not run on CUDA in this version. No forecast cost/throughput is measured here, and the earlier MedGemma resource grant is not reused. Keep full development/validation, bulk extraction and MRI work paused.

[Protocol](PROTOCOL.md) · [Reproduction](README.md) · [Software verification](aggregate/verification.json) · [Review record](REVIEW.md)

ChatGPT reviewed the tested implementation and found no required local corrections. The [review record](REVIEW.md) preserves the remaining CUDA-validation and independent output-token decoding checks for a separate execution stage. GitHub CI passed for code commit b9ceb21.
