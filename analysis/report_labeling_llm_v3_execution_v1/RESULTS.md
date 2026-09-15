# MedGemma v3 execution results

**The approved session is complete as an operational attempt, but the planned comparison is incomplete and failed its completion gate.** The control completed five reports; the evidence-first candidate completed three malformed responses and then hit the 4,096-token cap on its fourth report. Dispatch stopped automatically. No retries, remaining repeats, validation inference or training followed.

## Input → output → objective

Input was the same five original-language development reports (four English, one Spanish), with frozen definitions and one of two fixed prompts. No MRI images, organizer answers or study identifiers entered the prompts. The intended output was one JSON object containing twelve condition objects, each with `label`, `evidence_text` and `confidence`. Labels remain **positive / negative / uncertain / not_mentioned**; missing mentions are never converted to negative.

The objective was to test whether an evidence-first prompt reduced semantic contradictions while retaining usable structured output. The candidate arm failed the technical completion gate on the four attempted reports; its fifth report and second repeat were not run. It cannot establish semantic improvement, a paired winner or clinical accuracy.

## Recorded execution

| Run | Recorded / planned responses | Completed | Technical pass / recorded cells | Technical pass / planned cells | Binary proposals |
|---|---:|---:|---:|---:|---:|
| Control 1 | 5/5 | 5 | 48/60 | 48/60 | 34 |
| Candidate 1 | 4/5 | 3 | 0/48 | 0/60 | 0 |
| Candidate 2 | 0/5 | NOT RUN | — | 0/60 | — |
| Control 2 | 0/5 | NOT RUN | — | 0/60 | — |

Nine model calls produced durable records: eight completed responses and one truncated response. Eleven planned calls were not run. There are 108 recorded condition cells out of 240 planned; 48 passed structural/evidence checks, 60 failed, and 132 have no recorded output. **Technical acceptance is not accuracy or permission to use a training label.** The independent sample remains five studies, not 108 or 240 independent observations.

All five control responses had a strict leading thought envelope removed before frozen validation; all then parsed as JSON. There were nine evidence-span errors and three ambiguous-evidence errors. Accepted states were 14 positive, 20 negative, zero uncertain and 14 not_mentioned. Thus binary proposal coverage was 34/60 (56.7%); none has completed qualified semantic review.

The candidate's first three responses were syntactically valid JSON but used string values instead of the required nested condition objects. This produced 36 schema errors, with no accepted evidence/confidence records. Its fourth response generated 4,096 tokens in 211.55 seconds and was marked `generation_truncated`; all twelve cells were rejected before parsing. The 300-second timer did not cause this stop. The fifth candidate input and both second repeats were not dispatched. No malformed output was repaired or relabeled.

## Per-condition control results

The candidate had zero technical passes for every condition: three schema failures and one truncation per condition, plus one unrun report. The table below describes the five control outputs only; these are not ground-truth comparisons.

| Condition | Technical pass | Positive | Negative | Not mentioned | Technical failure |
|---|---:|---:|---:|---:|---:|
| ACL | 4/5 | 2 | 2 | 0 | 1 |
| MCL | 4/5 | 1 | 3 | 0 | 1 |
| Medial Meniscus | 5/5 | 1 | 4 | 0 | 0 |
| Lateral Meniscus | 4/5 | 2 | 1 | 1 | 1 |
| Medial OA | 5/5 | 1 | 1 | 3 | 0 |
| Lateral OA | 4/5 | 0 | 1 | 3 | 1 |
| PF OA | 4/5 | 1 | 1 | 2 | 1 |
| Effusion | 3/5 | 3 | 0 | 0 | 2 |
| Synovitis | 2/5 | 2 | 0 | 0 | 3 |
| Baker's | 4/5 | 0 | 2 | 2 | 1 |
| Contusion | 4/5 | 1 | 1 | 2 | 1 |
| Fracture | 5/5 | 0 | 4 | 1 | 0 |

[Full per-run/per-condition counts](aggregate/execution_per_condition.csv) retain all planned denominators. No uncertainty state was accepted in this attempt; this is an output distribution, not evidence that the reports lacked uncertainty.

## Language, agreement and earlier benchmarks

Control technical acceptance was 39/48 English cells and 9/12 Spanish cells. Candidate acceptance was 0/36 recorded English cells and 0/12 Spanish cells. Candidate English coverage was incomplete. One Spanish study and four English studies do not constitute multilingual validation; no translation was used.

Repeat consistency is **not evaluable** because neither second repeat ran. No usable binary candidate/control agreement or ensemble analysis is possible because the candidate produced no technically accepted labels. Organizer-reference agreement and semantic accuracy are **not evaluated**, rather than zero. The existing complete-session gate correctly refuses the paired blinded-review bundle; the separate partial diagnostic does not bypass it.

The [frozen rule baseline](../report_labeling/RESULTS.md), [earlier MedGemma/Qwen benchmark](../report_labeling_llm_v1/RESULTS.md) and [v2 smoke](../report_labeling_llm_v2/RTX_SMOKE_RESULTS.md) remain historical context. Qwen and the rule extractor were not rerun here. Their different recipes and/or cohorts must not be pooled into a new head-to-head accuracy ranking. No tuning on the 18 validation labels occurred, and the exact existing 40/18 split is unchanged. That small, previously inspected holdout still cannot support strong per-condition reliability claims.

## Runtime, cost and recovery

The session lasted **785.39 seconds (13.09 minutes)** including model loads/cache checks. Generation time was 524.46 seconds for control 1 and 233.63 seconds for candidate 1. Peak allocated GPU memory was **8.52 GiB**. Detailed exact model, pins, prompts, hardware and operational authorizations are in [EXECUTION.md](EXECUTION.md); the execution source and plan were not changed.

All **28 remote files** were copied locally and checked against the final remote SHA-256 inventory before shutdown. Inventory SHA-256: `19f31b50f361f911cc3f139f8a96dddadd0cfd377a3450644f1368bccbf83ea7`. Raw outputs, input/prompt tokens, original reports, grants, watchdog receipt, setup logs and provider stop evidence remain private. The original prepared directory and historical outputs were preserved separately. A new private migration archive containing the executed package, final dashboard and secret-free operator provenance was restored and verified across 45 files. The archive supplements earlier backups; it does not contain model weights or the MRI dataset.

All **450 CPU tests passed**, including six dashboard integrity/privacy checks. The original control-run verifier passed; the complete-session verifier correctly rejected the partial session. All 156 protected historical analysis files and the original prepared package remain byte-identical. [Verification record](aggregate/execution_verification.json).

The operator stopped the GPU at approximately **09:05:19 UTC**, before the original 09:30 deadline. An independent provider query confirmed EXITED; the console showed compute/container **Not running** and **$0.00/hour**. The watchdog was armed, but its scheduled CLI stop did not fire because the operator stopped early. The temporary account control key was **disabled**, and its local secret file removed; the disabled provider audit entry remains.

Using start-request-to-stop time, the replacement cost estimate is **about $0.34**; including the earlier approximately $0.10 setup attempt gives **about $0.44 total**, within the approved $2. These are rate/time estimates, not an invoice; allocation delay and billing settlement can change the final charge. Existing unrelated storage charges are separate.

## Recommendation

**Do not scale or treat this candidate as an improvement.** Preserve this failed attempt. The next local investigation should reconstruct saved input/output tokens with pinned tokenizer assets, audit the failed candidate child receipt, and separate the short schema failures from the long truncated response. These additional forensic checks have not yet been performed; the current diagnostic verifies inventory bytes, prepared-plan binding and raw-response reparse, plus the completed control verifier. Any revised prompt/output contract needs a separately versioned prospective experiment; do not loosen this run's parser, select favorable responses, or automatically extend the token budget. Qualified review is still required for the control's technically accepted labels. No full-40 development inference, 18-study validation run, bulk extraction, MRI training or YOLO work is authorized by these results.

The [local dashboard](diagnostics/README.md) visualizes the exact original-language input, raw output, evidence checks and unrun cells. It is a technical operator view with visible arms, not blinded adjudication. The public [aggregate execution record](aggregate/execution_outcome.json) contains no report text, study IDs, credentials or weights. The earlier [setup-only failure](SETUP_ATTEMPT.md) and [control preflight](CONTROL_PREFLIGHT.md) remain separate historical records.
