# Move to another GPU provider

There are two independent things to preserve:

1. **Code and experiment record:** clone this public GitHub repository.
2. **Private state:** transfer a verified backup of `state/`. GitHub intentionally excludes it.

## Current recovery point

The local project contains `state/runs/20260913T135620Z/` with the trained MedGemma adapter, processor, 15 validation predictions before and after training, the original 43/15 split, all 90 rendered validation input images, and 43 training-study teacher targets. Existing student outputs are preserved under `state/runs/student-pilot26/` while that work is paused.

`backups/medgemma-pilot-migration.tar.gz` preserves these artifacts and the competition CSV metadata/range manifest. It has been restored into a temporary directory and verified against all recorded SHA-256 checksums.

**This backup does not contain the raw DICOM dataset or base MedGemma weights.** The 3.68 GB selected DICOM subset remains on the stopped RunPod network volume and can also be re-downloaded with authorized Kaggle access. Base weights can be re-downloaded at the pinned Hugging Face revision. No passwords, refresh tokens, or SSH keys are included. Deleting the old network volume is a separate decision; this workflow does not delete it.

The old run has no optimizer or RNG checkpoint. Its adapter is sufficient for inference and initializing a new training run. Exact continuation of that historical optimizer trajectory is unavailable. New-format checkpoints support continuation from the last completed save.

## Create a backup before leaving a server

The analysis packaging also adds `runs/report-labeling-20260913-v1/` and `runs/label-audit-20260913/` to private state. These preserve original fingerprints, fixed splits, all gold-study extractions, report evidence and local analysis documents. The bundle command below includes them automatically. The older `medgemma-pilot-migration.tar.gz` predates these analyses; use the newer `rsna-knee-with-analyses-20260914.tar.gz` recovery point or create a fresh bundle.

After restoring, `python analysis/report_labeling/reproduce.py` uses `RSNA_STATE_DIR` to verify the frozen report experiment and write a new private result directory. No provider-specific settings or GPU are needed. GitHub contains only source and aggregate results, so preserve this private backup as well.

Wait until an atomic checkpoint completes; pause training before copying the rest of a changing run directory. Then:

```bash
python -m rsna_knee.bundle create \
  --state-dir "$RSNA_STATE_DIR" \
  --output /private-backups/rsna-knee-state.tar.gz
```

Add `--include-images` to include raw DICOMs under `state/data/`. Model caches are always excluded because they can be downloaded again. With a full competition dataset, use a dedicated private storage transfer instead of a huge single archive. The script refuses to overwrite an existing archive and records a checksum for every file.

Copy the archive using your authenticated SSH/SFTP or private storage connection. For example, using a host already configured in your personal SSH config:

```bash
scp /private-backups/rsna-knee-state.tar.gz gpu-next:/mnt/persistent/
```

SSH hosts, ports, and private keys belong in your personal SSH configuration or provider settings. They are not part of this repository. No fixed IP, RunPod pod ID, or `/workspace` path is needed by the package.

## Restore on the next server

Clone the repository and install the GPU environment as described in the README. Choose a **new, nonexistent directory** for restored state:

```bash
python -m rsna_knee.bundle restore \
  --archive /mnt/persistent/rsna-knee-state.tar.gz \
  --state-dir /mnt/persistent/rsna-knee-restored
export RSNA_STATE_DIR=/mnt/persistent/rsna-knee-restored
rsna-knee doctor
```

The restore rejects traversal paths, symlinks, duplicate archive entries, missing files, and checksum mismatches. It never merges over an existing directory.

If DICOMs were omitted, restore them over SSH or run `python scripts/download_data.py --data-dir "$RSNA_STATE_DIR/data"`. Use existing Kaggle access; the downloader can prompt for a token without echoing it. Authenticate for gated model downloads through the normal Hugging Face login or an environment secret. Never transfer credentials through Git.

For a future checkpoint, restore its **whole run directory**, including `development_split.csv`, `ground_truth.csv`, `input_fingerprints.csv`, `validation_before.csv`, and `checkpoints/`. Check out the same code revision recorded in `provenance.json`. Then use `rsna-knee train --resume .../checkpoints --max-steps N` with `N` as the total desired number of optimizer steps. A fresh output directory records the resumed session.

## What is portable, and what is not

| Item | Where preserved | Recovery |
|---|---|---|
| Code, pinned model revision, core dependency versions | Public GitHub | Clone and install |
| Trained adapter + processor | Private state/backup | Restore and load with base model |
| Split, predictions, labels, viewer inputs | Private state/backup | Restore |
| Future optimizer and RNG state | Private checkpoint | Restore with matching code/config/input fingerprint |
| Historical pilot optimizer state | Never recorded | Start a new optimizer from the adapter |
| DICOMs | Original volume or optional full backup | Copy or re-download authorized pilot selection |
| Base model cache | Original volume; excluded from default backup | Re-download pinned revision |
| Credentials and SSH access | Personal secret storage | Reconfigure on the new provider |

RunPod compute was stopped after the pilot. Storage still incurs its existing charge. None of these scripts starts a provider instance, purchases compute, submits to Kaggle, or uploads a model to a hub.
