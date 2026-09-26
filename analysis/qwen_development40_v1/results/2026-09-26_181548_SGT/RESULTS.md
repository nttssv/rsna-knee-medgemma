# Qwen baseline-A development benchmark — 26 September 2026

This is the aggregate record for the real one-pass development benchmark. It
used the frozen 40-report prepared package, control prompt A, Qwen/Qwen3-14B at
revision `40c069824f4251a91eefaf281ebe4c544efd3e18`, BF16, SDPA, greedy
decoding, batch size 1 and the unchanged primary-v2/secondary-v1 parsers.
Organizer labels were absent from model inputs. No validation inference,
training, repair generation or retry occurred.

## Execution identity

- Source commit: `f718f094985649628e41e063c2ec41c5d46f6453`
- Execution-plan SHA-256: `4af4d19ecfdd612ce96ab04c9760f1fb874b6506dd2df5e368ef4c8ccf940b52`
- Prepared-plan SHA-256: `b2c6f7e4d220ab83f5556170334fa441dac28c495e3ab9e9b8fefc8e956ca023`
- Resource: one Secure Cloud NVIDIA RTX A6000, 80 GB container disk, no attached persistent or network volume
- Signed-in rates: $0.53/hour compute plus $0.011/hour storage
- Provider session: 46 minutes 13.83 seconds from container PID 1 start to accepted stop request
- Estimated session cost: $0.4168 at the observed rates; this is an estimate, not an invoice

## Results

| Measure | Result |
|---|---:|
| Planned / attempted / completed / failed / unrun generations | 40 / 40 / 40 / 0 / 0 |
| Raw / parsed generation records | 40 / 40 |
| Planned condition rows | 480 |
| Technically valid rows | 442 (92.08%) |
| `ambiguous_evidence_error` rows | 21 |
| `evidence_error` rows | 17 |
| Reports with at least one parser technical-failure record | 18 |
| Inference elapsed time | 1,142.35 s (19 min 2.35 s) |
| Per-report generation time, min / median / max | 19.12 / 27.33 / 59.10 s |
| Input tokens, min / median / max | 992 / 1,302 / 2,600 |
| Output tokens, min / median / max | 385 / 542.5 / 1,175 |
| Peak allocated GPU memory | 28.29 GiB |

Labels among the 442 technically valid primary rows were 194
`not_mentioned`, 99 `negative`, 118 `positive` and 31 `uncertain`. These are
extractor outputs, not medical-accuracy measurements.

Rows with technical failures occurred in ACL (1), Baker's (1), Contusion (2),
Effusion (8), Fracture (1), Lateral Meniscus (4), Lateral OA (2), MCL (5),
Medial Meniscus (7), Medial OA (1), PF OA (4) and Synovitis (2). Counts here are
condition rows; the durable failure ledger has 18 report-level records because
one report can contain more than one affected condition.

## Artifact and shutdown evidence

- Private output archive SHA-256:
  `8282ebb04f33c6f86c05169d462441c094b5112197250eb739283493975f1f53`
- Archive transfer check: passed.
- Internal artifact hashes verified locally: 10/10.
- Provider stop mutation returned the exact pod with desired state `EXITED`.
- Signed-in console then showed compute and storage not running and total
  `$0.00/hour`.
- The final temporary GraphQL control key was disabled after the stop check.

Raw reports, prompts, responses, output token IDs, parsed evidence, study IDs,
session authorization, resource observation and shutdown receipts remain in the
private ignored run directory. They are intentionally absent from this public
repository.
