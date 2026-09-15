# Runtime provenance and advisory review

This version implements the next step explicitly authorized by the user after the Qwen local preparation. It preserves that preparation and every earlier analysis file.

The supervisor/receipt/review scaffolding was copied from the protected [v3 local runtime](../report_labeling_llm_v3_runtime_v1/README.md) and adapted in this new directory. It imports the frozen [v2 process supervisor](../report_labeling_llm_v2/scripts/run_smoke.py), offline Qwen tokenizer/backend/cache verifier, and unchanged v1/v2 parsers. No historical module/configuration globals are patched. The new source is independently fingerprinted.

Qwen-specific changes include the pinned Qwen candidate inputs/prompts and native template, its EOS IDs and2048-token cap, both parser records, exact primary repeat identity, durable pre-call attempt ledgers, incomplete-run diagnostic accounting, and isolation of organizer-reference data from runtime/review. Unlike the old review finalizer, this version does not open organizer labels at all; it produces only a clearly deferred comparison template.

The previous ChatGPT review requested this as a separate bounded runner/review milestone. ChatGPT reviewed public commit [b9ceb21](https://github.com/nttssv/rsna-knee-medgemma/commit/b9ceb21d1b76262582b8b844b302f64f2006dbea) in the existing project conversation and identified **no required local correction**. The exact code commit passed GitHub CI. It considered the hard execution guard, fixed inputs/prompts, worker provenance, raw durability, both parser checks, primary repeat identity and isolated semantic-review workflow adequate for this LOCAL milestone.

Two requirements remain for the separately versioned execution stage: verify the actual CUDA/cache/BF16/SDPA/stopping/memory/runtime/provider-shutdown behavior, and preferably add an independent saved-output-token decode roundtrip using the pinned tokenizer. The current adapter constructs raw text from generated IDs, but offline completed-run verification checks the saved IDs/counts/EOS and reparse rather than independently decoding those IDs. This is a stated verification limit, not a detected mismatch.

No prompts, parsers, gates or runtime code changed after that review. This advisory review is not independent execution, clinical adjudication, execution approval or verification of GPU behavior.
