# Completed pilot and current limits

Date: 2026-09-13. Original run: `20260913T135620Z`.

| Item | Recorded value |
|---|---|
| Teacher | google/medgemma-1.5-4b-it |
| Base revision | 91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b |
| GPU | A100 PCIe 80 GB |
| Training / validation studies | 43 / 15 |
| Input | 2 slices per sagittal/coronal/axial plane, 448px RGB |
| Precision | NF4 double quantization; BF16 autocast |
| Trainable weights | Language q/k/v/o attention LoRA, rank 8 |
| Optimizer | AdamW, learning rate 5e-5 |
| Updates / accumulation | 20 / 4 |
| Training time | 178.881 seconds |
| Peak allocated training GPU memory | 20.036 GiB |
| Mean validation AUC, original / adapted | 0.646283 / 0.658072 |

The CSVs in `experiments/2026-09-13-medgemma-pilot/` contain actual aggregate measurements. Per-study identifiers and predictions remain in private state.

The 4,407-study training set had only 58 fully organizer-labeled studies at the time of the audit. Missing labels were not converted to negatives. The pilot used a fixed group split based on normalized duplicate reports and usable patient IDs scoped to scanner metadata. Seven scanner groups overlap between training and validation; this is not independent-site validation. Reports were used only to group duplicates, never as teacher input.

The output is the relative next-token score for Yes versus No for each of 12 condition prompts. No radiology reports, bounding boxes, or segmentation masks were generated. Scores are not calibrated probabilities. A 0.50 cutoff is illustrative; AUC uses rankings. Six sampled slices can miss relevant anatomy, while the ground-truth label applies to the full study.

The adapted teacher's 43 training-study soft targets are **in-sample**, not out-of-fold predictions. They must not be described as independent predictions. The saved student experiment did not improve with distillation (mean AUC 0.615959 labels-only versus 0.607098 distilled); its image encoder was frozen. Further student development is paused by user request.

## Packaging validation

The preprocessing function bodies match the reference notebook exactly. CPU checks cover directory relocation, fixed split preservation, image fingerprints, unknown-label preservation, optimizer/RNG recovery against an uninterrupted update, incomplete checkpoint handling, and archive integrity/traversal checks. They use synthetic MRI data and a small CPU model, requiring no competition data or Hugging Face access in GitHub Actions.

The portable training implementation preserves the pilot model, prompts, quantization, loss, and sampling strategy, while adding configurable paths and atomic checkpoints. It has not yet been run end to end on CUDA. Before a larger run, perform a bounded GPU smoke test and verify prediction parity against the saved adapter on the fixed holdout. The assembly task did not restart a GPU or incur new training charges.

## Next session

The later report-labeling experiment is recorded in [analysis/report_labeling/RESULTS.md](../analysis/report_labeling/RESULTS.md). Its frozen v1 rule extractor made 92/216 held-out binary decisions, with 87 correct and five incorrect. The recommendation is more manual validation before scaling; it does not authorize further MRI training or labeling the remaining 4,349 reports. Its split is 40/18 and distinct from the MRI pilot's 43/15 split. All detailed report evidence is private state.

1. Read this file and `docs/MIGRATION.md`.
2. Inspect `state/runs/20260913T135620Z` and the case viewer before proposing further training.
3. Keep the current MedGemma focus; do not resume YOLO work without user direction.
4. Preserve the validation split and compare any new recipe against the existing saved results.
5. When moving to a new GPU provider, clone the private repository and restore private state before downloading anything large.
