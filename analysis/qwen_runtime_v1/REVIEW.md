# Runtime provenance and advisory review

This version implements the next step explicitly authorized by the user after the Qwen local preparation. It preserves that preparation and every earlier analysis file.

The supervisor/receipt/review scaffolding was copied from the protected [v3 local runtime](../report_labeling_llm_v3_runtime_v1/README.md) and adapted in this new directory. It imports the frozen [v2 process supervisor](../report_labeling_llm_v2/scripts/run_smoke.py), offline Qwen tokenizer/backend/cache verifier, and unchanged v1/v2 parsers. No historical module/configuration globals are patched. The new source is independently fingerprinted.

Qwen-specific changes include the pinned Qwen candidate inputs/prompts and native template, its EOS IDs and2048-token cap, both parser records, exact primary repeat identity, durable pre-call attempt ledgers, incomplete-run diagnostic accounting, and isolation of organizer-reference data from runtime/review. Unlike the old review finalizer, this version does not open organizer labels at all; it produces only a clearly deferred comparison template.

The previous ChatGPT review requested this as a separate bounded runner/review milestone. Advisory review of this published implementation will be recorded after the local checks and publication. This review is not clinical adjudication, execution approval or verification of GPU behavior.
