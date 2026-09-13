# Report-labeling results — frozen v1

**Recommendation C: more validation and manual annotation before scaling.**

On 18 held-out studies, the rule-based extractor made 92 binary decisions across 216 condition checks (42.6% coverage): 87 correct, five incorrect. It abstained on 124 checks: 20 uncertain and 104 with no supported mention matched. Conditional micro accuracy was 94.6%, but only **40.3% of all checks received a correct binary label**. The independent unit is a study, not each of the 216 correlated condition cells.

| Condition | Decided / 18 | TP | FP | TN | FN | Uncertain | No match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ACL | 10 | 4 | 0 | 6 | 0 | 2 | 6 |
| MCL | 11 | 0 | 0 | 9 | 2 | 1 | 6 |
| Medial Meniscus | 12 | 6 | 0 | 6 | 0 | 0 | 6 |
| Lateral Meniscus | 11 | 5 | 1 | 5 | 0 | 1 | 6 |
| Medial OA | 7 | 1 | 0 | 6 | 0 | 2 | 9 |
| Lateral OA | 5 | 1 | 0 | 4 | 0 | 2 | 11 |
| PF OA | 8 | 1 | 0 | 7 | 0 | 2 | 8 |
| Effusion | 8 | 2 | 0 | 6 | 0 | 5 | 5 |
| Synovitis | 4 | 4 | 0 | 0 | 0 | 0 | 14 |
| Baker's | 3 | 0 | 0 | 3 | 0 | 4 | 11 |
| Contusion | 5 | 3 | 0 | 2 | 0 | 1 | 12 |
| Fracture | 8 | 3 | 2 | 3 | 0 | 0 | 10 |

Macro balanced accuracy is 92.2% over ten conditions with defined values; macro recall is 90.9% over eleven. These are conditional on making a binary decision. Full sensitivity, specificity, PPV, NPV, F1, accuracy, kappa, prevalence, missingness and Wilson intervals are in [validation_metrics.csv](aggregate/validation_metrics.csv). [Macro metrics](aggregate/validation_macro_metrics.csv) report how many conditions contributed.

## Findings that limit scaling

- Two false-negative MCL extractions came from a broad sprain rule that incorrectly treated grade-2 injury as low grade. MCL had no true positives in the held-out set.
- A lateral-meniscus false positive came from missing the uncertainty synonym `suspicious`.
- Two fracture false positives were affirmative report statements that disagreed with the organizer reference. Acute-fracture definitions, timing and report/image agreement need radiologist review; this does not establish that the reference is wrong.
- Three Bulgarian reports, absent from development, caused 36 unsupported-language abstentions. The development sample included seven languages; there was no held-out Croatian, Dutch or German report.
- Other limitations include loss of prose compartment context, missing synonyms, unquantified severity, postoperative ambiguity and genuine apparent report omissions. Several clear report/reference disagreements were also observed in development.

Explicit ACL and medial-meniscus statements are the most promising candidates for additional validation, but PPV was based on only 4/4 and 6/6 positive predictions (Wilson lower bounds 0.51 and 0.61). OA apparent perfect accuracy involved only one true positive per compartment. Baker's had no positive decisions; Synovitis had no negative decisions.

The fixed confidence ≥0.90 tier retained 88 checks and all five errors. Confidence thresholds alone have not been shown to solve these failures. No condition passed the predeclared conservative scaling gate; an 18-study holdout cannot certify both PPV and NPV lower bounds above 0.80 even with perfect predictions.

**Do not convert not_mentioned to negative.** Every condition had gold positives among no-match outputs; some reflect unsupported language or failed terminology, while reviewed apparent omissions also include positive reference labels. [Nonmention analysis](aggregate/nonmention_analysis.csv) distinguishes those categories without altering predictions.

## Aggregate artifacts

- [High-confidence metrics](aggregate/validation_high_confidence_metrics.csv)
- [Development metrics](aggregate/development_metrics.csv), reported separately from validation
- [Fixed split prevalence balance](aggregate/split_balance.csv)
- [Post-evaluation error-category counts](aggregate/error_category_counts.csv)
- [Candidate label-yield projections](aggregate/label_yield_estimates.csv)

Yield projections extrapolate holdout acceptance rates to 4,349 studies. They assume a representative language/site distribution, which was not verified, and their intervals cover sampling variation only. They are **not generated labels or quality-approved usable counts**. None of the 4,349 studies was processed by the extractor.

The baseline remains unchanged after validation. Future repairs need separately versioned code and fresh independent validation; replaying v1 verifies reproducibility only. A useful next step is manual review spanning rare positives, languages, severity boundaries and report/reference disagreements, followed by condition-specific acceptance criteria. No further MRI training is justified by this feasibility result alone.

[Organizer definitions](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343), [dataset description](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data).
