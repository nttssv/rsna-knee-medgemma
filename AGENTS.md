# Project continuity

- Keep this GitHub repository private. Do not change visibility.
- Read README.md, docs/EXPERIMENT.md, and docs/MIGRATION.md before starting work.
- Current focus: MedGemma input/output review and teacher improvement. YOLO work is paused.
- The completed pilot is preserved under state/runs/20260913T135620Z locally; it is not in GitHub. Inspect it before assuming training must start over.
- The original adapter supports inference and warm-start training, not exact optimizer resume. New-format checkpoints include full training state.
- Keep credentials, competition metadata, per-study IDs, MRI files, and model weights out of Git. Use state/ and a private backup.
- Use RSNA_STATE_DIR for persistent paths. Do not embed provider IPs, pod IDs, SSH keys, usernames, or mount paths in training code.
- CPU audits and checks are local and do not require starting a GPU. Do not provision paid compute implicitly.
- Before committing, run relevant tests and python scripts/check_staged.py after staging explicit source paths.
- Do not overwrite an existing run. Preserve split and input fingerprints when comparing experiments.
- The scientific claims are limited to a small feasibility pilot. Scores are uncalibrated Yes/No rankings; case labels describe full studies, not the sampled slices alone.
- For print requests, default to duplex and black-and-white unless the user specifies otherwise.
