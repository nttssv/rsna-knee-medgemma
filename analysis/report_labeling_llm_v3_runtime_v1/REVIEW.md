# Runtime integration review

This is a local software milestone following the advisory review of the frozen v3 candidate. It preserves the prompts, definitions, five reports, ABBA order, shared 4096-token cap and strict response framing. No advisory response authorizes paid compute or qualifies as clinical adjudication.

The implementation separates the frozen design from a new runtime package. Its execution guard deliberately refuses even with `--execute`. Synthetic CPU checks exercise bounded subprocess dispatch, durable raw receipts, completed-session verification and blinded review collection. A later source revision must explicitly address resource authorization and provider shutdown before real generation.

Source review status will be recorded after local verification. The scope of any browser review is the published source and aggregate software checks, with no private reports or study identifiers transmitted.
