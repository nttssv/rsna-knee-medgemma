# Qwen evidence-selection A/B — real saved outputs

Run: 2026-09-25; five development reports; no organizer comparison.

| Block | Generations | Rows | Valid | Technical failures | + | − | ? | Not mentioned | Binary | Seconds | Peak GiB |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| A1 | 5 | 60 | 58 | {'ambiguous_evidence_error': 2} | 16 | 19 | 1 | 22 | 35 | 158.33 | 28.01 |
| B1 | 5 | 60 | 58 | {'ambiguous_evidence_error': 2} | 13 | 19 | 1 | 25 | 32 | 157.24 | 28.08 |
| B2 | 5 | 60 | 58 | {'ambiguous_evidence_error': 2} | 13 | 19 | 1 | 25 | 32 | 162.04 | 28.08 |
| A2 | 5 | 60 | 58 | {'ambiguous_evidence_error': 2} | 16 | 19 | 1 | 22 | 35 | 171.53 | 28.08 |

Label counts and binary decisions include only technically accepted primary-v2 rows. Raw proposals are separately preserved in summary.json and condition_rows.csv. Technical failures are never converted to negative.

## Repeatability

- A: raw byte-identical 5/5; protocol-identical 60/60; valid-and-identical 58/60; fixed 60/60 gate **FAIL**.
- B: raw byte-identical 5/5; protocol-identical 60/60; valid-and-identical 58/60; fixed 60/60 gate **FAIL**.

## A/B differences

- A1_vs_B1: 16/60 raw proposals differ; 15/60 primary parsed rows differ (label/evidence/confidence/status).
- A2_vs_B2: 16/60 raw proposals differ; 15/60 primary parsed rows differ (label/evidence/confidence/status).

Technical validity is not medical accuracy. Confidence is model self-report. Semantic comments remain provisional; no adjudication is resolved.


## Execution and shutdown

- Source commit: `1ab7b5bbc9638fc81eac17ec52e3a104c56d622a`.
- Execution-plan SHA: `6d551f7bf19db648e11eed9766e6a47b984567e6b8fbc6e4dd660c738bf9c54a`.
- Model: Qwen/Qwen3-14B at `40c069824f4251a91eefaf281ebe4c544efd3e18`; BF16, SDPA, greedy, non-thinking, batch 1; no training.
- Five fixed development reports. Planned 20 / attempted 20 / generation-completed 20 / incomplete-or-integrity-failed 0 / unrun 0. All 20 raw and parsed records saved. Eight condition-level technical failures are separate from generation failure counts.
- First durable attempt: 2026-09-25 11:56:08 UTC (19:56:08 Singapore); final generation finished 12:06:58 UTC (20:06:58 Singapore). Generation loop elapsed 649.23 seconds; supervisor hard limit 1,799 seconds.
- A40, driver 570.195.03, torch 2.8.0+cu128, CUDA 12.8; native BF16; peak allocation 28.0821 GiB. Pinned package, full cache/checksum and input-token parity checks passed; inference offline.
- Download plus full cache audit took 60.26 seconds; 13 verified files / 29,548,135,507 bytes. These timings are measured for this host, not a future guarantee.
- All 33 files copied and SHA-256 verified before stop. Archive SHA: `5cb8edc9ba31a00c8bfe67866e24713f08a02a016b7cf58a098919f3e251a519`.
- Provider stop accepted at 12:07:48 UTC (20:07:48 Singapore); independent REST read showed EXITED and console showed compute/storage Not running, total $0.00/hour. Temporary control key Disabled afterward; external watchdog cancelled only after stop verification.
- Successful pod cost estimate $0.1672; prior selections within the same clock $0.1627; cumulative session estimate **$0.3299**. This is elapsed time × displayed rates, not an invoice. Earlier completed session approximately $0.41 is recorded separately, not erased. Original T0/deadline and cumulative $3 budget were preserved.

## Interpretation and next step

Raw-proposal differences are **16/60**, while primary parsed-row differences are **15/60** for both A/B repeat pairings. Four labels change, one evidence quote changes, and ten accepted differences are confidence-only. The extra raw-only confidence change belongs to a rejected ambiguous-evidence row. These are this new A/B run's numbers; historical snapshot 2f6604bb remains 20/60 raw versus 18/60 parsed.

Secondary v1 diagnostic parser: A1/A2 each 58 valid + 2 evidence_error; B1/B2 each 59 valid + 1 evidence_error. It does not replace primary-v2 judgments.

**One recommendation: fix target-specific bone-contusion evidence/negation before choosing B.** B still uses muscle contusion for the bone target and misses explicit bone-negation; its fracture-polarity change and patellofemoral omission need review. B is not selected for expansion. The private local SEMANTIC_REVIEW.md and semantic_review_all_240.csv contain provisional screening of every cell; neither is published. No prompt/parser was changed, no organizer comparison performed, and no old adjudication resolved.


Private original reports, study IDs, raw/token records, row-level outputs, semantic checklist, credentials and model cache are excluded from this publication. Scientific source/configs/prompts/parsers and historical outputs are unchanged.
