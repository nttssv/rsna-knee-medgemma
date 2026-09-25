# Qwen extraction results — 2026-09-25 17:49:55 Asia/Singapore

Qwen3-14B completed the planned 20 generations over five frozen development reports in control-1 → candidate-1 → candidate-2 → control-2 order. No training, validation inference, prompt changes, repair, or generation retries occurred.

| Block | Generations planned / attempted / completed / failed / unrun | Condition rows | Technically valid | Evidence errors | Other technical failures | Positive | Negative | Uncertain | Not mentioned | Binary decisions | Generation seconds (sum) | Peak allocated GPU memory (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control-1 | 5 / 5 / 5 / 0 / 0 | 60 | 58 | 2 | 0 | 16 | 19 | 1 | 22 | 35 | 147.45 | 28.01 |
| candidate-1 | 5 / 5 / 5 / 0 / 0 | 60 | 58 | 2 | 0 | 11 | 16 | 3 | 28 | 27 | 127.51 | 28.13 |
| candidate-2 | 5 / 5 / 5 / 0 / 0 | 60 | 58 | 2 | 0 | 11 | 16 | 3 | 28 | 27 | 127.09 | 28.13 |
| control-2 | 5 / 5 / 5 / 0 / 0 | 60 | 58 | 2 | 0 | 16 | 19 | 1 | 22 | 35 | 147.73 | 28.13 |

Across the run, 20/20 generations completed and 232/240 condition rows were technically valid. The eight invalid rows reflect two repeated ambiguous-evidence cells in each of four blocks; their accepted labels remain null. Technical validity does not measure medical accuracy.

## Repeatability

| Prompt repeats | Raw responses byte-identical | Primary parser protocol fields identical | Both repeats valid | Frozen gate |
|---|---:|---:|---:|---|
| Control | 5/5 reports | 60/60 condition rows | 58/60 rows | FAIL |
| Candidate | 5/5 reports | 60/60 condition rows | 58/60 rows | FAIL |

Both prompts were exactly repeatable at the raw-response and parsed-field levels. The frozen repeatability gate still fails for both prompts because it requires all 60 condition rows to be technically valid in each repeat.

## Prompt comparison and review status

For the representative first control and candidate passes, 20/60 structured condition outputs differed in at least one field: nine label differences, 12 evidence-text differences, and 17 confidence differences (overlapping categories). The underlying cases require semantic review. The automated review identified a repeated target-scope concern around muscle-only contusion being labeled as a bone contusion. This is a review recommendation, not clinical adjudication; no item is marked resolved.

Organizer/reference labels were not opened or compared. Confidence is the model's self-report and is not calibrated. The five-report development sample is too small to support a medical-accuracy or scaling claim. No prompt is selected from this run.

## Model and execution provenance

- Model: `Qwen/Qwen3-14B`, revision `40c069824f4251a91eefaf281ebe4c544efd3e18`
- Runtime: BF16, SDPA, greedy decoding, batch size 1, output cap 2048
- Environment: NVIDIA RTX A6000; Torch 2.8.0+cu128; CUDA runtime 12.8
- Verified pinned cache: all 13 files, 29,548,135,507 bytes; exact input token parity and independent output-token decoding passed
- Executed operator-plan SHA-256: `c8dee4efd82d6a4a7bd89ad1e6910d64cc32c52296c83f183a66b57624cf59b6`
- Local complete archive SHA-256: `dab478b4e68e5db30a76533bb5fd9d2a61ea935772eb37af28508f537cf71116`

The local execution record reports approximately $0.61 estimated cumulative cost for the approved session, including an earlier unsuccessful segment; this is not a reconciled invoice. It separately records approximately $0.41 for an older session. The provider console was observed at $0.00/hour after stop. These are operator-recorded observations, not independently reconciled billing data.

Study-level outputs and original reports remain local under ignored `state/runs/qwen-extraction-a6000-20260925/` and were not copied into this public snapshot.
