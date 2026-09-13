# Dataset label audit — 2026-09-13

The accessible competition archive contains exactly **4,407 training studies and 3 example-test studies** (4,410 unique studies), 24,386 MRI series and 819,635 DICOM files. The hidden scoring set is outside these exact counts.

Exactly five non-DICOM files were found: `train.csv`, `train_series.csv`, `test.csv`, `test_series.csv`, and `sample_submission.csv`. No additional JSON, Excel, text or other standalone annotation file was present. Every study/series table identifier matched its corresponding image folder; no extra diagnostic table or failed join was found.

`train.csv` contains 58 studies with all twelve organizer labels, 4,349 studies with all twelve condition fields blank, and a nonempty report for every training study. There are no partially labeled studies. The 58 were identified by selecting rows with any nonmissing condition value, after numeric/binary validation. `sample_submission.csv` contains 0.5 placeholders, not ground truth. Series fields describe acquisition, not disease.

The original MRI pilot took its condition ground truth from those `train.csv` cells, split into 43 training and 15 validation studies. The independent report-labeling experiment uses the same 58 reference studies with its own fixed 40/18 split.

The complete archive directory and all five CSVs were inspected. DICOM diagnostic-text checks covered 1,172 locally available files from 15 studies, **not all 819,635 headers**; the sampled fields revealed no additional diagnostic annotations. Source CSV hashes matched the pilot copies.

Tracked aggregate evidence:

- [Condition label counts](aggregate/condition_label_counts.csv)
- [Column missingness](aggregate/column_missingness.csv)
- [Study join counts](aggregate/study_join_checks.csv)
- [Input file fingerprints](aggregate/metadata_file_inventory.csv)

The complete local report, per-study coverage and 58 labeled identifiers are preserved under `$RSNA_STATE_DIR/runs/label-audit-20260913/`. Original source metadata is under `$RSNA_STATE_DIR/data/`. These are private state, not Git content.

[Organizer dataset description](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data).
