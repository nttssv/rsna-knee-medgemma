# Project continuity

- This GitHub repository is public by user request on 2026-09-14. Keep private data, credentials and model artifacts excluded from Git.
- Read README.md, docs/EXPERIMENT.md, and docs/MIGRATION.md before starting work.
- Current focus: MedGemma input/output review and teacher improvement. YOLO work is paused.
- Read analysis/README.md and analysis/report_labeling/RESULTS.md before further label or model work. MRI training and bulk extraction on the 4,349 reports remain paused.
- The separate analysis/report_labeling_llm_v1/ benchmark completed five development reports twice/model at executed commit f334958. Both all-valid repeatability gates failed. Read README.md, RESULTS.md, REVIEW.md and RESOURCES.md. RunPod was stopped; no full 40-study development or 18-study validation LLM inference or bulk labeling occurred. Preserve the execution fingerprints and exact split. Any further GPU stage needs a concrete reviewed version and current user resource authorization.
- Keep the report-labeling v1 source/config/tests byte-identical to analysis/report_labeling/frozen_code_manifest.json. Its 18-study holdout has been inspected; make future fixes in a separately versioned experiment with a new validation plan.
- Per-study report-label outputs and evidence belong in ignored state/runs/report-labeling-20260913-v1, never in analysis/ or Git. Only aggregate findings are committed.
- The completed pilot is preserved under state/runs/20260913T135620Z locally; it is not in GitHub. Inspect it before assuming training must start over.
- The original adapter supports inference and warm-start training, not exact optimizer resume. New-format checkpoints include full training state.
- Keep credentials, competition metadata, per-study IDs, MRI files, and model weights out of Git. Use state/ and a private backup.
- Use RSNA_STATE_DIR for persistent paths. Do not embed provider IPs, pod IDs, SSH keys, usernames, or mount paths in training code.
- CPU audits and checks are local and do not require starting a GPU. Do not provision paid compute implicitly.
- Before committing, run relevant tests and python scripts/check_staged.py after staging explicit source paths.
- Do not overwrite an existing run. Preserve split and input fingerprints when comparing experiments.
- The scientific claims are limited to a small feasibility pilot. Scores are uncalibrated Yes/No rankings; case labels describe full studies, not the sampled slices alone.
- For print requests, default to duplex and black-and-white unless the user specifies otherwise.
