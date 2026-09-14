# Results and execution status

**MedGemma and Qwen have NOT RUN. There are no LLM validation scores yet.** This release publishes a reproducible benchmark implementation and its local checks. Model downloads, tokenization with model tokenizers, actual inference, prompt development on model outputs and the execution freeze remain pending approval.

## What is observed

The existing source and original split fingerprints were checked. Local preparation produced report-only inputs for exactly **40 development and 18 validation studies**, with no organizer-label fields in inference files. Both model commands passed dry-run checks. The frozen rule source and existing split were preserved. Tests exercise synthetic outputs, including the complete prepare/freeze/evaluate path; synthetic metrics are not research results and are not published as model performance.

| Extractor | Evaluation evidence | Binary decisions | Correct | Incorrect | Abstentions/failures | Correct yield over all checks |
|---|---|---:|---:|---:|---:|---:|
| Frozen rule v1 | Previously completed, 18 studies / 216 checks | 92 | 87 | 5 | 124 semantic abstentions | 40.3% |
| MedGemma 1.5 4B | NOT RUN | — | — | — | — | — |
| Qwen3 14B | NOT RUN | — | — | — | — | — |

Rule v1's coverage is 42.6% and conditional micro accuracy is 94.6%. Its five errors and 124 abstentions remain the reference. Full baseline per-condition metrics, denominators and intervals are in [baseline_reference_metrics.csv](aggregate/baseline_reference_metrics.csv), copied from the previously published aggregate. The LLM table deliberately has no invented zeros or placeholder scores.

## Per-condition status

| Condition | Rule v1 decided / 18 | Rule TP / FP / TN / FN | MedGemma / Qwen | Current scaling status |
|---|---:|---|---|---|
| ACL | 10 | 4 / 0 / 6 / 0 | Not run | Requires fresh validation |
| MCL | 11 | 0 / 0 / 9 / 2 | Not run | Manual review required |
| Medial Meniscus | 12 | 6 / 0 / 6 / 0 | Not run | Requires fresh validation |
| Lateral Meniscus | 11 | 5 / 1 / 5 / 0 | Not run | Manual review required |
| Medial OA | 7 | 1 / 0 / 6 / 0 | Not run | Not established |
| Lateral OA | 5 | 1 / 0 / 4 / 0 | Not run | Not established |
| PF OA | 8 | 1 / 0 / 7 / 0 | Not run | Not established |
| Effusion | 8 | 2 / 0 / 6 / 0 | Not run | Not established |
| Synovitis | 4 | 4 / 0 / 0 / 0 | Not run | Not established |
| Baker's | 3 | 0 / 0 / 3 / 0 | Not run | Not established |
| Contusion | 5 | 3 / 0 / 2 / 0 | Not run | Not established |
| Fracture | 8 | 3 / 2 / 3 / 0 | Not run | Report/reference review required |

These status notes come from the existing rule baseline, not from MedGemma or Qwen predictions. The promising ACL/medial-meniscus counts are too small to establish safe supervision. No condition currently has evidence that either LLM or an ensemble is suitable for high-confidence weak labels.

## Pending comparisons

- **Model metrics:** condition-level sensitivity, specificity, PPV, NPV, F1, accuracy, balanced accuracy, meaningful kappa, coverage, state counts, technical failures and correct yield.
- **Agreement:** five distinct binary acceptance strategies, plus descriptive matching-abstention counts. No empirical ensemble performance exists yet.
- **Language:** descriptive results using the original language review, including three Bulgarian validation reports and explicit absence of Croatian/Dutch/German. No evidence yet shows an LLM improves multilingual extraction.
- **Errors:** compare negation, uncertainty, anatomy, severity, timing, postoperative findings, omissions and report/reference disagreements after execution. No LLM fixes or new failure categories have been clinically adjudicated.
- **Resources:** [RESOURCES.md](RESOURCES.md) separates observed configuration from estimated VRAM/time. Actual LLM runtime and peak VRAM are unmeasured.

## Recommendation

**E — more manual annotation and fresh independent validation are required before scaling.** Complete a bounded development smoke test after user approval, then assess whether a full exploratory benchmark is worth running. Keep the remaining 4,349 reports unprocessed and MRI training paused.

The reused 18-study set cannot provide fresh validation, model self-confidence is uncalibrated, and matching model outputs do not create independent ground truth. Model size, possible public-data pretraining exposure, report/reference ambiguity and tiny language/condition denominators limit interpretation. A future result can nominate a candidate for fresh adjudicated evaluation; it cannot retroactively validate this already-inspected holdout.
