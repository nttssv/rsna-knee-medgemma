# Private T4 replica throughput diagnostic

Based on completed Version 8 snapshot
`4a4382ab8546cd872e90a3e81654291be9fa7546`.
This package changes scheduling only. The original T4 package and its
**KAGGLE_RUN_PASS** result remain unchanged.

Two new Python processes receive `CUDA_VISIBLE_DEVICES=0` and `=1`
respectively before Torch is imported. Each validates exactly one visible
Tesla T4, loads the same frozen base/adapter on its local `cuda:0`, and runs
the original single-GPU implementation. This is independent replication,
not model parallelism. No fallback or retry exists, including after OOM.

The unchanged Version 8 source/config hashes are checked in both parent and
worker. NF4/double quantization, FP16 linear compute, FP32 residual policy,
all six images, processor, vision scheduling, target prompts, twelve forwards
per study and score computation remain byte-identical to that version.
Both workers retain the same two-forward first-study diagnostic before their
assigned prediction studies; those extra diagnostics are recorded as overhead.

Study IDs are sorted lexically and assigned alternately to workers 0/1. The
three visible studies therefore form shards of two and one. Parent merging
restores the original `test.csv` order and applies the unchanged validator.
Both workers must exit successfully with matching PID, shard, device and
checksum receipts. One failure kills/reaps the sibling, preserves partial
outputs/logs and prevents final `submission.csv` publication.

## Fixed parity gate

All 36 scores are compared with the exact private downloaded Version 8 CSV,
SHA-256 `1f3d823c4b1e689b252c041945440dcc3685d0e5810cb7fbd91583739ef739cd`.
The threshold is fixed prospectively at **absolute difference <= 1e-6**,
relative tolerance **0**. Exact-equality count and maximum absolute difference
are reported. Every ID/column must match; nonfinite values fail. Any score
difference above the threshold stops publication and requires investigation;
there is no score correction or favorable selection.

The reference contains image-model predictions, not organizer labels. It is
embedded only in the private diagnostic notebook for parent-side comparison
after inference. It never enters a model prompt. This example-only notebook
intentionally refuses a larger hidden-test set that does not match the
three-study reference. It is not a scored competition submission notebook.

## Boundaries and measurements

The user requested this diagnostic; it retains the latest 120-minute maximum.
The notebook starts a shared 115-minute deadline before asset audit/install,
with five minutes reserved for shutdown/artifact collection. The operator
also monitors the external 120-minute wall/quota ceiling. No competition
submission, paid service, training, Qwen run or model change is included.
Notebook and assets remain Private, Internet OFF, offline package pins intact.

Artifacts include worker stdout/stderr, load and complete-study timings,
actual dtype/device maps, per-GPU peak allocated/reserved memory, independent
worker status, and nvidia-smi utilization samples every approximately two
seconds. Sampling failure is recorded as unavailable, never filled with zero.
Process lifetime and study-work fractions are reported separately from GPU
utilization percentages. Kaggle quota is recorded from its actual UI before
and after the run, not inferred by doubling wall time.

Throughput is measured using complete studies. The small two/one split causes
load imbalance; report both observed example wall time and the explicitly
estimated balanced long-run throughput. Two replicas cannot be assumed to
make the prior approximately 34.5-hour hidden estimate fit nine hours.
Only real example completion plus the fixed parity gate earns
**THROUGHPUT_READY**. It does not establish medical accuracy, BF16 parity,
hidden-test runtime readiness, SUBMITTED or SCORED.

## Reproduce local preparation

```bash
.venv/bin/python -m pytest -q analysis/kaggle_image_t4_replicas_v1/tests \
  analysis/kaggle_image_t4_v1/tests
.venv/bin/python analysis/kaggle_image_t4_replicas_v1/scripts/build_notebook.py \
  --reference-csv state/kaggle_image_t4_v1/runs/20260926T151922Z/outputs/submission.csv \
  --output state/kaggle_image_t4_replicas_v1/private_notebook/dual_replicas_diagnostic.ipynb
```

The builder refuses an existing output notebook. In the existing private
Kaggle editor, import it, attach the competition and unchanged private assets,
verify T4 x2/Internet OFF/Private, then Save & Run All once. Download outputs,
check the parent parity/status and both worker logs, and confirm Draft Session
off and no active run. Never submit this diagnostic to the competition.
