# Runtime integration review

This is a local software milestone following the advisory review of the frozen v3 candidate. It preserves the prompts, definitions, five reports, ABBA order, shared 4096-token cap and strict response framing. No advisory response authorizes paid compute or qualifies as clinical adjudication.

The implementation separates the frozen design from a new runtime package. Its execution guard deliberately refuses even with `--execute`. Synthetic CPU checks exercise bounded subprocess dispatch, durable raw receipts, completed-session verification and blinded review collection. A later source revision must explicitly address resource authorization and provider shutdown before real generation.

ChatGPT reviewed published source commit `401bdf71c2849444fed2669ea6dd11a530beffb5` and judged the bounded runtime complete. It identified one narrow review-software issue: forcing technically invalid cells into semantic categories could confound technical failure with semantic error.

The correction introduces `technical_failure_unreviewable` solely as a review disposition. Invalid cells require exactly this disposition plus notes; valid cells prohibit it and require the original semantic categories. The interface disables inappropriate choices. Final outputs retain technical status separately, count the semantic-review denominator only for technically valid cells, and never add technical dispositions to semantic-error categories. Seven additional checks cover exclusivity, invalid/valid-cell handling and mixed-status denominators. No frozen prompt, target definition or extraction state changed.

Fresh runtime plans and demo outputs bind the corrected fingerprints; the earlier preparation and receipts are preserved and are not silently rewritten. The source review explicitly left CUDA execution, current resource authorization, provider shutdown and qualified clinical adjudication to later stages. No private reports or study identifiers were transmitted for this advisory review.
