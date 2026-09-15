# Qwen local audit and next experiment

**Qwen is now the report-extraction focus; MedGemma work is paused. No new Qwen inference has occurred.** This milestone audits saved outputs and prepares a concrete prospective comparison.

## What the saved evidence shows

The same five development reports (four English, one Spanish) were processed twice in the original Qwen smoke. Both original prediction hashes, run/config/source manifests, case sidecars and reparsed outputs matched. All 189 protected historical analysis files are unchanged.

| Historical Qwen measure | Repeat 1 | Repeat 2 |
|---|---:|---:|
| Completed reports | 5/5 | 5/5 |
| Original technical passes | 58/60 | 58/60 |
| Original quotation failures | 2/60 | 2/60 |
| Total output tokens | 2,605 | 2,605 |
| Mean generation seconds/report | 22.69 | 25.34 |

All five raw response strings are identical across repeats. Only 58/60 condition outputs are both valid and identical; matching failures do not pass the existing repeatability gate. These are historical A100 timings, not new measurements or RTX 6000 Ada throughput predictions.

## Parser policy is not model improvement

Both failed quotations have unique matches after the existing v2 ASCII-whitespace normalization. However, v2 also requires unique source evidence and rejects two other formerly accepted quotations that appear multiple times. All four affected cells are in the one Spanish report. A single case cannot establish a language effect.

| Historical v1 status → diagnostic v2 status | Cells per repeat |
|---|---:|
| valid → valid | 56 |
| evidence_error → valid (whitespace-only match) | 2 |
| valid → ambiguous_evidence_error (repeated exact quote) | 2 |
| Total | 60 |

Thus the diagnostic acceptance count stays **58/60, with different accepted cells**. This is **historical raw-output re-evaluation under a prospective parser**, not new inference, corrected labels or replacement official metrics. No new organizer scores were calculated. An exact or whitespace-equivalent source quote still does not prove correct anatomy or interpretation: one recovered quotation was already flagged for a compartment concern.

## What remains unresolved

The inherited 12-item review queue is preserved: two quotation failures, nine binary organizer discordances and one questionable uncertain output. Its tentative category counts are: four report/reference disagreements, three anatomy/compartment concerns, two verbatim-evidence issues, one threshold/reference issue, one contradictory-report/timing issue and one unspecified-severity issue. These are analyst hypotheses, not twelve proven model errors or clinical adjudications. The two new parser-ambiguity flags overlap existing review items; no new adjudicated labels were created.

The historical 26/35 binary organizer matches versus rule v1's 28/34 remain descriptive five-study results. Their denominators differ, and the original acceptance policies differ. No clinical ranking follows. [Original results](../report_labeling_llm_v1/RESULTS.md) remain authoritative. Report/evidence review comes before organizer agreement.

## Prepared prospective input and output

**Input:** original report text, in its original language, plus either the exact historical v1 instructions/definitions or the existing unrun v2 instructions/definition clarifications. Neither arm includes images, study IDs, organizer labels or another model's answers.

**Output:** the same 12 condition keys, each containing a four-state label, evidence quotation and uncalibrated confidence. Silence remains `not_mentioned`; it never becomes negative.

Both packages use `Qwen/Qwen3-14B` at the same pinned revision, with thinking disabled. CPU tokenization used four checksum-verified cached assets and the ten pinned dependency versions. All ten historical rendered control prompts and their input counts matched. Original per-call token IDs were not saved, so no original-token-ID roundtrip is claimed.

| CPU input sizing | Control (v1) | Candidate (existing v2) |
|---|---:|---:|
| Minimum tokens/report | 1,277 | 1,670 |
| Maximum tokens/report | 1,631 | 2,024 |
| Total across five reports | 6,833 | 8,798 |
| Largest input + 2,048 output reserve | 3,679 | 4,072 |

Both fit the 8,192 input cap and saved 40,960 config context limit. This demonstrates sizing only; no weights or GPU were used. The candidate is 393 tokens longer for each tested report. Its extra instructions/clarifications are documented in the [complete prompt diff](configs/prompt_diff.json); it is a whole-package comparison, not a single-factor test.

## Decision

Proceed to review a separately implemented, bounded **Qwen-only 20-generation smoke**: five known development reports × two instruction packages × two repeats, with the same primary parser for both. The local [protocol](PROTOCOL.md) retains the strict 60/60 valid-repeat advancement gate and requires qualified semantic review. A 58/60 technical count is not permission to scale.

The proposed smoke is NOT RUN and has no execution adapter in this package. Future generation needs a concrete resource plan and current authorization; no paid action was taken here. Full development, reused validation, bulk extraction and image training remain paused. Fresh adjudication and independent evaluation remain necessary beyond these five inspected cases.

## Verification

All **482 CPU tests passed**, including 20 new synthetic audit/contract checks. They test corruption rejection, preservation of historical outputs, parser-policy differences and exact prompt reuse; they do not measure Qwen's medical performance. Public [aggregate evidence](aggregate/summary.json) records the source/protocol/config hashes and actual tokenizer versions. [README](README.md) gives reproduction instructions.
