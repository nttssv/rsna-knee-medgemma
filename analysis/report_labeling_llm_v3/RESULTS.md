# Local v3 results — no generation

**Status: prepared and tokenized; no model outputs or performance metrics exist.** The evidence-first prompt is a prospective hypothesis, not a demonstrated improvement.

All **296 repository tests passed**, including 44 new checks. The new checks cover historical/control/definition preservation, shared strict-envelope behavior, rejection of malformed envelopes, truncation precedence, explicit demonstration that lexical checks cannot fix medical meaning, serialization of 24 synthetic expectations, local-only preparation, blank review templates, no-overwrite and changed-source/split/manifest rejection. The synthetic cases do not prove how MedGemma responds.

## Measured CPU tokenization

All ten cached tokenizer/processor/config assets and all ten runtime package pins passed verification. Real tokenization of both prompts on the same five reports gave:

| Arm | Reports | Input-token range | Largest input + 4,096 reserved output |
|---|---:|---:|---:|
| v3-control | 5 | 1,618–1,945 | 6,041 |
| Evidence-first candidate | 5 | 1,880–2,207 | 6,303 |

Both pass the 8,192 input limit and pinned 131,072 context limit. The candidate adds 262 input tokens per report. This sizing does not establish generation completion, runtime, memory use or semantic quality. Processor deprecation warnings were observed; rendered-text/token parity passed. [Public tokenizer receipt](aggregate/tokenizer_preflight.json) records versions, provenance and counts; rendered report prompts and input IDs are private.

Private design SHA-256: `feb7a4a5a6132f1d97e1fd470f0b0195f0b92f6f0bc42136a15e0a77c678b775`. It specifies the four balanced arm/repeat runs and a 20-generation ceiling, but is marked DRY_DESIGN_ONLY with execution disallowed. A 240-row semantic-review template is blank; the 12 inherited adjudication items remain unresolved.

## What remains unknown

No v3 technical acceptance rate, semantic error rate, four-state distribution, organizer agreement, repeatability or per-condition performance exists. No validation, full-development or multilingual performance claim is possible. One Spanish and four English known development reports are not independent evidence of generalization; the synthetic French/Spanish cases are authored expectations only.

The candidate and control share the new framing and output cap. Any future paired result concerns the prompt formulation under those common settings, not a causal claim about framing versus token budget or a direct rerun of historical v2. There is no GPU adapter in this local package. Runtime integration, output attestation, qualified semantic review, a current resource quote and explicit authorization remain prerequisites to an actual paid stage.

All 124 protected historical tracked files—including both v1 folders, executed v2 code and the response diagnostic—remain unchanged. The fixed five inputs and original 40/18 split retain their fingerprints. No old predictions were rescored, no not_mentioned labels became negatives, and no new training targets were generated.

The private recovery archive was restored and verified across 530 files. All 523 pre-existing non-cache private files remained byte-identical. Publication checks covered Markdown links, staged secrets/private artifacts and the Git diff.
