# SUBMISSION_READY — frozen V10 submission artifact

Prepared locally on **2026-09-27 SGT**, from
`26dafd65b91e054577110f4c695f7d72a7660faf`.
**SUBMISSION_READY describes this packaged, contract-tested artifact.**
It does not mean a new GPU run, hidden-test completion, competition submission
or leaderboard score. No external execution or competition submission occurred
in this preparation task; historical V10 results remain unchanged.

## Artifact and exact change

The private notebook is
`state/kaggle_image_t4_submission_v1/private_notebook/submission.ipynb`.
SHA-256: **`8c69cef1508f25cd946217692ee3dc719b860491bd0751406d0f2b1eded7366d`**.
Its twelve cells compile and embed seven verified source/config/dependency
files. The notebook is kept out of the public repository; rebuild it with the
command in [README.md](README.md).

The new dispatch file removes the reference-score gate, visible-ID dependence,
two repeated diagnostic forwards/activation trace, and completed <40-second
benchmark gate. It obtains actual test IDs dynamically, balances sorted IDs
across two independent workers, and merges in original test order. The frozen
constructor replaces the diagnostic-loading helper; it still loads once in
the same single-T4 placement and retains dtype/free-memory checks. No retry,
fallback device map or extra scoring forward is introduced.

Source/asset hashes, native-logit/FP32-probability finiteness, cache accounting,
twelve ordered target records per study, exact completeness, schema, worker
failure propagation, supervised deadline and atomic final publication remain.
The existing utilization sampler remains. An empty shard is explicitly recorded
without scoring, loading a model, or manufacturing output rows. The future
production deadline is 530 minutes from setup, rather than the former
25-minute example diagnostic limit; it does not authorize a run now.

Frozen files are reused byte-for-byte: the three V8 numerical/preprocessing/
inference modules, configuration, requirements and V10 vision cache. Base
`google/medgemma-1.5-4b-it` revision
`91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`, LoRA adapter, NF4/FP16 compute,
existing FP32 residual/post-normalization policy, six-image preprocessing,
twelve prompts and No/Yes scoring are unchanged. No optimizer, Qwen, labels,
new training data or further throughput optimization is involved.

## Three-study output preservation: what was proved

The local replay consumed the **real saved V10** worker prediction files and
all 36 native scoring records, then invoked the same `merge_predictions` and
frozen submission validator used by this new runner. It produced a final CSV
**byte-identical to V10**, with all 36 scores unchanged and maximum score
difference **0**. Output SHA-256:
`8b66aef38061782800d830dd1c4bce4e21024a0ea34c6e404c98b9bafdb1a7ec`.

Private replay evidence is in
`state/kaggle_image_t4_submission_v1/v10_replay/replay_result.json`.
Mode: **LOCAL_REPLAY_NOT_NEW_INFERENCE**. The notebook contains neither the
replay helper, these predictions, nor any visible study ID. The frozen scoring
functions/cache and CPU dispatch tests provide source/contract evidence for
unchanged inference. **The new wrapper has not itself been rerun on a GPU**;
replay is not presented as fresh model inference. V10's original external
3-study/36-score exact parity result remains the actual GPU evidence.

## Runtime estimate from the completed V10 measurement

The [competition data page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)
was checked on 2026-09-27 SGT: it describes **about 1,300 test studies**.
The [code requirements](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview)
specify a **nine-hour GPU notebook limit** and Internet disabled.

| Estimate for 650 studies on the longer shard | Hours |
|---|---:|
| V10 mean 24.342 s/complete study | 4.40 |
| Slowest observed study, 25.802 s | 4.66 |
| Slowest observed × 1.20 plus ten-minute setup/IO allowance | **5.76** |

The estimate uses complete-study time, not a single forward. Formula for actual
count N: `ceil(N/2) * 25.801558146 * 1.20 + 600` seconds. N is **not hardcoded
in inference**. Hidden study size, transfer syntax, series length and IO tails
can differ from the three measured examples. The estimate supports feasibility,
not guaranteed completion. The model's prediction quality is unchanged and is
not established by throughput or replay parity. BF16 cross-precision parity
remains `NUMERICAL_PARITY_UNVERIFIED`.

## Privacy and execution boundary

The existing saved Kaggle notebook and model asset pages were refreshed and
observed **Private** on 2026-09-27 SGT. The editor showed **T4 x2**, **Internet
OFF**, **Draft Session off**; quota remained **00:44 / 30 hours**. The new
local notebook also declares Private/Internet OFF. It has not been uploaded,
Save & Run was not invoked, and no additional quota was consumed by this task.
After import, Kaggle can reset data attachments: reattach the same competition
and private model dataset and recheck those settings before an authorized run.

The original 32-file asset manifest remains
`4b4e9396301bd0c80a6a4ee1fd169e5b800d482f8833dcd28e3cf0b92ab4e0d7`;
adapter hash remains
`a0bbce8fade7c273dbbeeba39bf792f82eb166abf1bac3450d5100027496f448`.
The builder pins the original manifest and retains full per-file verification,
offline wheel installation and `local_files_only=True` model/processor loads.
Do not attach replay/reference CSVs or private review material.

## Completed local verification

**333 tests passed in 28.22 seconds:** 115 new submission contracts and 218
existing V10/V9/V8 tests. The new tests cover arbitrary test counts, balanced
shards and exact order, missing/duplicate/nonfinite/invalid worker outputs,
real CPU subprocess failure/timeout cleanup, no overwrite, source/cache hash
drift, and a direct worker test using the original prediction loop with a
synthetic backend. They prove twelve calls per study without invoking the
removed diagnostic, and exercise numerical/dtype/hardware gates. Synthetic
backend tests are not real model outputs. Notebook tests check byte-exact frozen
embedded sources, offline/private metadata and absence of prediction/ID inputs.

The original V8/V9/V10 directories have zero diff against the specified commit.
All eight frozen source bindings passed, including the original V10 dispatcher
used for the [reviewable dispatch diff](runtime_dispatch.diff). The source
manifest is [source_manifest.json](source_manifest.json), SHA-256
`657ddd697251c08f49102bf0888e7a37cc51f74a61cad13bd122aa1b07a7f277`.
New runner SHA-256:
`5c96fdce657f67f0bb8e60cc0c6fe7b5517646022aa69cff310b1d406365548c`.
Machine-readable [readiness checks](readiness_checks.json) distinguish the
local replay, prior GPU measurements, current privacy observation and absence
of new execution.

Execution and competition submission remain separate user decisions. The
prepared notebook has no API that submits itself, and no prior session approval
is treated as authorization for another GPU run.
