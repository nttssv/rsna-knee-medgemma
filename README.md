# RSNA Knee · MedGemma

Private research project for the RSNA Knee Abnormality Detection competition. The current focus is **MedGemma evaluation and improvement**. Further YOLO work is paused.

This repository contains the code, configuration, experiment record, and instructions. The separate `state/` directory holds data and trained models. Set `RSNA_STATE_DIR` to any persistent disk on any provider; no RunPod account or SDK is required.

## Current experiment

The dataset label audit and report-labeling analysis are now preserved under [analysis/](analysis/README.md). The report-only baseline used 40 development / 18 validation studies and remains frozen. Its current recommendation is **more validation/manual annotation before scaling**; MRI training and bulk report labeling remain paused. Full study-level analysis snapshots are in ignored `state/runs/`, with aggregate findings and reproducible code in Git.

The completed run `20260913T135620Z` used `google/medgemma-1.5-4b-it`, a pinned model revision, six MRI slices per study, and 20 QLoRA updates. It trained on 43 studies and evaluated 15 held-out studies. Mean AUC changed from **0.6463 to 0.6581**. This small pilot does not establish a reliable performance improvement.

The trained adapter, processor, fixed split, per-study predictions, case viewer, and teacher targets have been preserved locally under `state/runs/20260913T135620Z/`. A verified private migration archive is under `backups/`. These directories are excluded from Git.

**The original pilot did not save optimizer state.** It supports inference and starting further training with its learned adapter. Future runs from this project save full training checkpoints every five steps, including optimizer and random generator state.

## Folder map

```text
configs/pilot.toml                 Model revision and experiment settings
src/rsna_knee/                     Audit, MRI loading, training, checkpoints, viewer
scripts/bootstrap.sh              CPU or CUDA environment setup
scripts/download_data.py           Authorized, resumable pilot data download
docs/MIGRATION.md                 Move state to another GPU server
docs/EXPERIMENT.md                What was actually run and what is unverified
notebooks/                        Original pilot recipe, without saved outputs
experiments/2026-09-13-medgemma-pilot/  Aggregate measurements and training config
analysis/                        Dataset label audit and frozen report-labeling analysis
archive/student_pilot/             Preserved student code; development paused
tests/                            CPU portability and recovery checks
state/                            PRIVATE data, adapters, runs; ignored by Git
backups/                          PRIVATE migration archives; ignored by Git
```

## Set up on a GPU server

Use Linux, Python 3.12, and a CUDA GPU with native BF16 support. The original pilot ran on an A100 80 GB. The conservative memory guard is 37 GiB; smaller GPUs have not been validated. The NVIDIA driver must support the installed CUDA runtime.

```bash
git clone git@github.com:nttssv/rsna-knee-medgemma.git
cd rsna-knee-medgemma
export RSNA_STATE_DIR=/mnt/persistent/rsna-knee-state
bash scripts/bootstrap.sh gpu
source .venv/bin/activate
rsna-knee doctor
```

For a Mac or a CPU machine, use `bash scripts/bootstrap.sh cpu`. Audits and the result viewer work on CPU; this training recipe requires a CUDA GPU.

Existing Kaggle competition access and Hugging Face MedGemma access are required. Authenticate through your provider's secret manager, the normal Hugging Face CLI, or environment variables described in `.env.example`. Never commit tokens. This project does not accept model or competition terms on your behalf.

## Restore the completed work

Follow [MIGRATION.md](docs/MIGRATION.md). GitHub alone does **not** contain the trained weights or competition data. Copy the private migration archive as well, restore it into a new state directory, then restore or re-download the selected DICOMs.

To download the pilot data after gaining competition access:

```bash
python scripts/download_data.py --data-dir "$RSNA_STATE_DIR/data"
```

This fetches CSV metadata and selects the preferred series in each plane for organizer-labeled studies. It reads the ZIP directory and individual file ranges, verifies CRCs, and reuses verified files on retries. The original selection contains 5,637 DICOMs, about 3.68 GB extracted. The script refuses a selection exceeding 25 GB. It never fetches the full roughly 570 GB extracted dataset. The signed download URL stays in memory.

## Audit, train, and resume

All commands below create a new run directory by default. A run includes the split, ground truth without reports, input fingerprints, metrics, configuration, and completion status. Missing labels stay unknown.

```bash
# Audit without downloading model weights or training.
rsna-knee audit

# Start a new 20-step pilot. Save full checkpoints every five steps.
rsna-knee train

# Evaluate the completed legacy adapter on its original held-out split.
rsna-knee evaluate \
  --adapter "$RSNA_STATE_DIR/runs/20260913T135620Z/teacher_adapter" \
  --split-file "$RSNA_STATE_DIR/runs/20260913T135620Z/development_split.csv"

# Start further training from the legacy adapter, with a NEW optimizer.
# max-steps here means 20 additional updates in a new experiment.
rsna-knee train --max-steps 20 \
  --init-adapter "$RSNA_STATE_DIR/runs/20260913T135620Z/teacher_adapter" \
  --split-file "$RSNA_STATE_DIR/runs/20260913T135620Z/development_split.csv"

# Resume a NEW-FORMAT run at its last full checkpoint, to 40 total steps.
# Replace RUN_NAME with an actual run created by this package.
rsna-knee train --max-steps 40 \
  --resume "$RSNA_STATE_DIR/runs/RUN_NAME/checkpoints"
```

Resume uses the original split automatically and writes a new run directory. It checks configuration, labels, preprocessed images, and source fingerprints before applying optimizer state. Keep the same Git revision and configuration when migrating. Changing GPU type can change floating-point results; cross-GPU bitwise equivalence is not promised.

Use `--run-dir` after a subcommand to name a new output directory, or `--data-dir` to override the data path. Use `--config configs/local.toml` **before** the subcommand for a private configuration copy. No credentials belong in TOML files.

## Inspect inputs and outputs

The restored pilot viewer is ready without the full DICOM dataset or a GPU:

```bash
rsna-knee review --run-dir "$RSNA_STATE_DIR/runs/20260913T135620Z"
```

Open [localhost:7862](http://127.0.0.1:7862). For a new run, generate its viewer using `rsna-knee export-cases --run-dir /path/to/run` first. The viewer currently supports this six-image, 448px recipe. Ground truth covers the whole study; the input covers only sampled slices. Predictions are uncalibrated Yes/No ranking scores.

Training always writes `metrics.jsonl` and `training_history.csv`. To import a completed run into a local Trackio dashboard:

```bash
python -m rsna_knee.tracking import \
  --run-dir "$RSNA_STATE_DIR/runs/20260913T135620Z" \
  --tracking-dir "$RSNA_STATE_DIR/tracking"
python -m rsna_knee.tracking serve --tracking-dir "$RSNA_STATE_DIR/tracking"
```

Both servers bind to `127.0.0.1`. When running remotely, use SSH forwarding, for example `ssh -L 7862:127.0.0.1:7862 your-gpu-host`. No public dashboard is created.

## Development and validation

```bash
python -m pip install -e '.[test]'
python -m pytest -q
git config core.hooksPath .githooks
git add README.md configs src scripts tests docs notebooks experiments archive requirements .github .githooks pyproject.toml .gitignore .dockerignore .env.example Dockerfile AGENTS.md
python scripts/check_staged.py
git commit -m "Describe the change"
```

The CPU tests verify checkpoint/optimizer recovery, interrupted-save behavior, archive safety and integrity, unknown-label handling, split separation, directory relocation, and preprocessing parity with the recorded notebook. The refactored GPU path still needs a bounded CUDA smoke run before a longer experiment. No GPU was restarted to assemble this repository.

The default test run also includes the report extractor's semantic checks and analysis migration/fingerprint checks. To reproduce the frozen report analysis with restored private state, run `python analysis/report_labeling/reproduce.py`; see its [README](analysis/report_labeling/README.md).

The optional Dockerfile uses an official PyTorch CUDA base. The image tag was checked, but a container build and GPU execution were not performed as part of this packaging task. Dependency constraints record key pilot versions, not a complete historical environment lock.

## Access and licenses

Keep this repository **private**. Do not publish competition images, case identifiers, reports, tokens, or model checkpoints through GitHub. The code repository is not a substitute for a private artifact backup. Model access and upstream licenses remain applicable on a new provider; the PEFT adapter still requires the original base model. See the [MedGemma model card](https://huggingface.co/google/medgemma-1.5-4b-it), [PEFT checkpoint guide](https://huggingface.co/docs/peft/developer_guides/checkpoint), and [competition rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules).
