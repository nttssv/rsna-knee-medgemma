# Development adjudication queue

Preparation writes `v2_adjudication_candidates.csv` inside the selected private state output. It contains the same five development studies only, with original organizer label, report evidence/context, rule v1 and Qwen v1 output, Qwen status, a pointer note for rejected MedGemma raw output, reason/category, reviewer fields and status. A rejected MedGemma output is never treated as an accepted prediction.

The 12 review items comprise nine binary discordances, two lexical evidence failures and one questionable uncertain output. Their categories are tentative analyst hypotheses. There is no qualified human adjudication recorded in this task: **all remain unresolved**. Counts alone are published in [aggregate/adjudication_counts.csv](aggregate/adjudication_counts.csv).

Suggested statuses for a separate, qualified review:

- `unresolved`: no adjudication completed.
- `insufficient_information`: a qualified reviewer records that available evidence is insufficient.
- `report_supports_organizer`: reviewer documents report-based support for the reference.
- `report_supports_model`: reviewer documents report-based support for the extraction; this does not alone prove the image-level reference wrong.
- `genuinely_ambiguous`: conflicting or insufficient report evidence remains.
- `organizer_definition_requires_image_review`: the target cannot be resolved from text alone.

Record reviewer identity/qualification, date, rationale and evidence in private state when adjudication actually happens. Preserve the original candidate queue and write a separately versioned review. Do not automatically edit `train.csv`, overwrite organizer labels, or tune prompts to force a disputed reference answer. Adjudication is currently a queued human task, not a completed clinical review.
