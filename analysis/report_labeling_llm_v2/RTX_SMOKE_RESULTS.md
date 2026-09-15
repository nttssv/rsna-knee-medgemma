# RTX 6000 Ada generation smoke — incomplete

**The GPU worked, but the MedGemma output contract failed.** On 2026-09-15, three reports were attempted in the first MedGemma repeat. The third response reached the 2,048-token limit, triggering the existing fail-fast stop. No labels were accepted. The remaining two reports, MedGemma repeat 2 and both Qwen repeats were not run. The provider was stopped after verified result transfer. Estimated compute cost was **$0.14**.

## Objective and recipe

Test technical validity, evidence grounding and repeatability for extracting 12 condition states from the same five development reports. Input is original report text plus instructions and definitions; no MRI images, translations, identifiers or organizer answers enter the prompt. Organizer answers are separate comparison references. Output must follow the existing positive / negative / uncertain / not_mentioned schema, with evidence and confidence. Not-mentioned is never converted to negative; a technical failure has no accepted label.

Executed source: `f09a4ddc3b3a795f97766686adc05a74b75a1fcc`. Reviewed plan SHA-256: `74d6efcacf4b13d4b99a3cc29bb1027f94a24a4dd9e87f7e836f68ff4dfd75ca`.

- MedGemma: `google/medgemma-1.5-4b-it`, revision `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`.
- Planned Qwen comparator: `Qwen/Qwen3-14B`, revision `40c069824f4251a91eefaf281ebe4c544efd3e18`; generation NOT RUN in this session.
- Exact [prompts](prompts/medgemma_prompt_v2.txt), [definitions](prompts/target_definitions_v2.txt), [model settings](configs/medgemma.json) and [runtime pins](configs/runtime.json) remain unchanged.
- BF16, batch one, SDPA, greedy decoding, one beam, seed 20260914, input cap 8,192, output cap 2,048, generation timeout 300 seconds. Qwen's planned template disables thinking; no equivalent supported MedGemma control was claimed.
- No JSON substring rescue, reasoning-prefix stripping, repair generation, retry, validation-label tuning or reference changes.

The configs retain historical preparation status fields; actual execution status is in immutable private receipts and the [public result aggregate](aggregate/rtx_smoke.json). They are not edited after execution to change fingerprints.

## Hardware change and runtime

A100 restart/migration attempts could not obtain capacity. The user explicitly requested RTX 6000 Ada. It was unavailable in the old volume's region, so a new temporary pod was deployed in another region with a fresh pinned environment and cache. This is a documented change from the original cache-reuse proposal: pinned assets were downloaded during setup, then the generation worker ran offline and checksum-audited every required MedGemma cache file before loading. Both model snapshots were downloaded; Qwen was never dispatched. Existing volumes and stopped A100 pods were preserved.

| Item | Observed value |
|---|---:|
| GPU | NVIDIA RTX 6000 Ada Generation, 48 GB |
| Driver / CUDA runtime | 570.124.06 / 12.8 |
| Python / PyTorch | 3.12.3 / 2.8.0+cu128 |
| Native BF16 / runtime pins | Passed / all ten matched |
| Peak allocated GPU memory, first MedGemma run | 8.453 GiB |
| Child / parent runtime | 257.61 / 258.77 seconds |
| Provider start request → stopped verification | 05:00:00 → 05:09:59 UTC |
| Compute rate / estimated charge | $0.84/hour / $0.1397 |
| Temporary disk | 80 GB; $0.011/hour; estimated $0.0018 |
| New persistent volume | None |

Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, digest `sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35`. Cost estimates use start-request to stopped-verification elapsed time, not an invoice; existing storage charges are separate. The new pod was verified stopped at $0.00/hour. Its temporary cache was disposable. Full environment freeze, setup/download/generation logs, helper code and provider receipt are retained privately with the copied results.

## Measured outputs

| Attempt within MedGemma repeat 1 | Language | Input / output tokens | Generation seconds | Generation / parser outcome | Accepted conditions |
|---|---|---:|---:|---|---:|
| 1 | Spanish | 1,625 / 1,666 | 75.13 | Completed / parse_error | 0/12 |
| 2 | English | 1,618 / 1,752 | 78.38 | Completed / parse_error | 0/12 |
| 3 | English | 1,945 / 2,048 | 91.88 | generation_truncated | 0/12 |
| 4–5 | English | — | — | NOT RUN | — |

All three responses begin with an extra thought-channel marker and explanatory text. The first two eventually include JSON, but the whole response violates the predeclared bare-JSON/whole-outer-fence contract. The third is truncated. Model-visible output conventions need investigation; neither changing GPU nor silently stripping text would establish a valid completed benchmark. No CUDA error or OOM was recorded.

Denominators: **3 distinct attempted studies, 3/20 planned generations, 36 attempted condition cells**. First-repeat planning accounts for 24 parse errors + 12 truncated cells + 24 unattempted cells = 60. These are correlated outputs, not 36 independent studies. Every condition has two parse failures and one truncated failure; none has an accepted state. Conditional reference accuracy is **undefined**, not zero. Accepted four-state counts are all zero because of technical rejection, not because all conditions were absent.

Rule v1 remains visible beside organizer answers in the private viewer. A new rule-vs-MedGemma-vs-Qwen accuracy comparison, agreement analysis, ensemble or repeatability score cannot be computed from this incomplete run. Historical [v1 results](../report_labeling_llm_v1/RESULTS.md) remain separate. One Spanish and two English attempts cannot establish multilingual performance. No 18-study holdout was run or tuned on; even a future n=18 result would have substantial uncertainty and sparse per-condition support.

## Integrity and visualization

All **17 original remote artifacts** were copied and verified against their remote SHA-256 inventory before stopping the pod. The parent receipt correctly records `provider_stopped=false` because it was written before external shutdown; a separate later provider receipt records the verified stop. Neither receipt is rewritten to erase chronology.

The [partial diagnostic tool](diagnostics/README.md) verifies the pinned inventory, unchanged candidate/input preparation, and exact raw-response parsing before rendering a private interactive report. It shows original-language input, exact raw response, organizer Yes/No, frozen rule output and explicit failed/NOT RUN states. Its incomplete banner and absent accuracy are intentional. The original completed-four-run viewer and acceptance gate are untouched. Per-study text, outputs and references remain excluded from Git.

## Recommendation

Keep paid compute stopped. Investigate the pinned MedGemma template and documented response-channel controls locally, and formulate a separately versioned prospective recipe before considering another bounded run. Do not rescore these responses as primary accepted labels by stripping their prefixes. Resolve the 12 inherited semantic-review items with a qualified reviewer before expansion. No full-development, validation, bulk extraction, MRI training or YOLO run is justified by this failure.

## Publication checks

All 235 repository tests passed. The diagnostic verified the real saved artifact inventory and raw-response parsing; a modified raw file in a temporary copy was correctly rejected. Browser checks confirmed original-language input, all 12 reference rows, truncated-case selection and Qwen NOT RUN placeholders. All 70 frozen v1 files and 478 preceding private files verified unchanged; executed v2 code fingerprints still match. The new private archive was restored and verified across 517 files. No cache, weights or credentials are included in Git.
