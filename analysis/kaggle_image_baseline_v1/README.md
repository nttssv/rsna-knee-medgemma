# Kaggle image baseline v1

Status: **LOCAL_READY** as of 2026-09-26 18:48 SGT.

This package prepares the first image-only competition submission from the
completed MRI pilot at `state/runs/20260913T135620Z/`. It is based on repository
snapshot `9ef96c7bf199618753324d77c96b62f2b093354a`. The Qwen report-extraction
results and historical pilot outputs are unchanged.

No Kaggle GPU inference, submission, or scoring has been performed. The current
notebook accelerator selector offered **GPU T4 x2**. The frozen pilot requires
native BF16 and at least 37 GiB on GPU 0. A T4 has compute capability 7.5 and
does not satisfy native BF16; two devices are not treated as pooled memory.
The notebook therefore fails before model loading on the currently offered
accelerator. The guard is deliberately retained.

## Verified checkpoint

- Base model: `google/medgemma-1.5-4b-it`
- Base revision: `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`
- Adapter: rank-8 LoRA, 272 tensors (136 A and 136 B)
- Adapter bytes: 17,865,944
- Adapter SHA-256: `a0bbce8fade7c273dbbeeba39bf792f82eb166abf1bac3450d5100027496f448`
- Base weights: two Safetensors shards, 8,600,277,880 bytes total
- Resolved model class: `Gemma3ForConditionalGeneration`
- Resolved processor class: `Gemma3Processor`
- Answer token IDs: No `3771`, Yes `10784`
- Adapter target modules missing from the pinned base: zero

The private local audit result is
`state/kaggle_image_baseline_v1/checkpoint_audit.json`. It verifies artifact
structure, hashes, the pinned base revision manifest, tokenizer/processor
resolution, and adapter target compatibility. A full CUDA weight load has not
been run locally because the workstation has no CUDA GPU.

## Inference contract

The notebook accepts only:

- competition `test.csv`;
- competition `test_series.csv`;
- competition test DICOM directories;
- the private offline base model, processor, adapter, and wheelhouse.

It does not read `train.csv`, reports, organizer labels, Qwen outputs, review
tables, or development IDs. It discovers the actual test IDs at runtime and
writes columns in the exact order from `sample_submission.csv`:

`StudyInstanceUID`, ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA,
Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture.

The runtime preserves the pilot preprocessing and model recipe: two sampled
slices per sagittal/coronal/axial plane, 448-pixel inputs, NF4 weights with BF16
autocast, eager attention, device 0, and twelve sequential Yes/No forward passes
per study. Scores are the softmax probability of the Yes token against the No
token. They are image-model scores; no Qwen confidence is used.

An unscored study aborts the run. The code never replaces a failed inference
with 0.5. It retains partial files and records DICOM issues, timing, and peak GPU
allocation, but publishes `submission.csv` only after every actual test ID has
a finite score in `[0,1]` and the exact schema passes.

## Private offline artifacts

These paths are intentionally ignored by Git:

- notebook: `state/kaggle_image_baseline_v1/private_notebook/rsna_knee_medgemma_image_baseline.ipynb`
- upload root: `state/kaggle_image_baseline_v1/assets/`
- model: `assets/base_model/`
- adapter and processor: `assets/adapter/`
- Linux Python 3.11/3.12 wheels: `assets/wheels/`
- asset manifest: `assets/asset_manifest.json`

The asset manifest binds 32 files, including both model shards, adapter,
processor/tokenizer artifacts, and all offline wheels. The notebook verifies
every file's byte count and SHA-256 before installing or loading anything.
Its manifest SHA-256 is
`4b4e9396301bd0c80a6a4ee1fd169e5b800d482f8833dcd28e3cf0b92ab4e0d7`.
The pack is approximately 8.2 GiB and contains no credentials.
The generated notebook SHA-256 is
`f146f07b157ea24caacad25bb1a1fae368616991105b4b23dc132765aa5f3582`.

## Build and validate locally

```bash
.venv/bin/python analysis/kaggle_image_baseline_v1/scripts/build_notebook.py \
  --output state/kaggle_image_baseline_v1/private_notebook/rsna_knee_medgemma_image_baseline.ipynb

.venv/bin/python -m pytest -q analysis/kaggle_image_baseline_v1/tests
```

The visible example metadata contains three test studies and the expected
submission schema, but the example test DICOM directory is not present in the
local workspace. No model-derived example `submission.csv` has therefore been
created.

## Required Kaggle attachments

1. The RSNA Knee Abnormality Detection competition input.
2. One private Kaggle dataset/model containing the contents of
   `state/kaggle_image_baseline_v1/assets/`, with `asset_manifest.json` at its
   root.
3. The generated private notebook.

Internet must remain off. The notebook installs dependencies only from its
attached wheelhouse and loads the model with `local_files_only=True`.

## Concrete blocker and minimal next path

The current T4 x2 notebook shape cannot run the frozen BF16/37-GiB/device-0
recipe. Proceed only if Kaggle exposes one native-BF16 GPU with at least 37 GiB,
or review a separately versioned inference adaptation for available T4
hardware. Such an adaptation would necessarily change precision and/or device
placement and must be tested against the frozen pilot before it can replace
this baseline runtime.

The full hidden test is approximately 1,300 studies. This code performs twelve
model forwards per study, so the hidden run is about 15,600 forwards. No
compatible Kaggle run has yet measured load time, per-study throughput, peak
memory, or the hidden-run safety margin.

## Status definitions

- **LOCAL_READY**: source, private assets, notebook, and CPU contract checks are complete.
- **KAGGLE_RUN_PASS**: a compatible Kaggle GPU ran the example test and produced a validated model-derived CSV.
- **SUBMITTED**: the validated notebook output was actually submitted.
- **SCORED**: Kaggle returned a competition score.

Only **LOCAL_READY** has been achieved.
