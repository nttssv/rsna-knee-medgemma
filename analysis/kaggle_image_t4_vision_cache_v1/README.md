# Same-study vision cache: target below 40 seconds

Status: **LOCAL_READY; GPU measurements pending.** This is a separately
versioned optimization of the completed [two-replica diagnostic](../kaggle_image_t4_replicas_v1/README.md).
Historical Version 8 remains **KAGGLE_RUN_PASS** and Version 9 remains
**THROUGHPUT_READY**. Neither result establishes this candidate's performance.

The goal is strictly **<40 seconds for every complete study on each T4**,
including image preprocessing and all twelve target forwards. It is not a
40-second individual forward or an average obtained by dividing two workers'
wall time. Version 9 measured approximately 93–99 seconds per study.

## Only change: reuse the identical vision output within one study

The frozen Gemma3 runtime invokes its vision encoder on the same six images
for every target. This adapter memoizes the output of the existing
`OrderedVisionMicrobatch` wrapper after the first target, then reuses it for
the other eleven. The original one-image vision scheduling, image order,
processor, projector, twelve decoder forwards, target prompts, No/Yes
scoring and submission validator remain unchanged. No attention, dtype,
quantization, checkpoint, adapter or weight changes are made.

The cache binds an explicit study identity and checks processed pixel bytes,
shape, stride, layout, dtype, device, keyword arguments, autocast context and
frozen weight identities/mutation versions on every call. A same-study drift
fails rather than silently recomputing. Only eval + inference mode is allowed.
Each returned tensor is cloned so a caller cannot modify future hits.
The cache is released on a study change, completion or failure. There are no
cross-study hits, retry, alternate placement, CPU/disk offload or training.

The original repeated-input numerical diagnostic runs before the cache is
installed. Image preprocessing is then cleared so all timed studies begin
cold. For each study the worker must report exactly one cache miss, eleven
hits and twelve scoring forwards. The parent verifies those receipts.
CPU tests use a tiny synthetic SigLIP model; they are not MedGemma results.

The saved hidden tensor is approximately 54 MiB for the current six-image
FP16 shape, plus approximately 28 MiB of reference pixels. Actual T4 memory
and speed remain unmeasured. Reducing 96 seconds to 40 requires saving over
56 seconds; removing eleven vision executions can only achieve that if vision
costs enough of the existing runtime. The diagnostic measures that component
with CUDA events instead of assuming a speedup.

## Prospective acceptance and run boundary

- Exactly the three visible test studies, deterministic two/one sharding,
  one independent model replica per T4, unchanged private offline assets.
- All 36 scores compared with the exact Version 8 reference CSV, SHA-256
  `1f3d823c4b1e689b252c041945440dcc3685d0e5810cb7fbd91583739ef739cd`.
  Absolute tolerance remains **1e-6**, relative tolerance **0**, fixed before
  this run. Report exact-match count and maximum absolute difference. The
  reference is parent-side model output, never a model input or a label.
- Every complete-study duration must be finite, positive and **strictly
  below 40.0 seconds** to receive `UNDER40_PARITY_PASS`. If example/parity
  passes but timing does not, status is `PARITY_PASS_TARGET_NOT_MET`.
- One worker failure or parity failure prevents the final combined CSV.
  Partial logs/results remain available. No fallback scores or retry.
- Proposed new diagnostic: one session, **30 minutes maximum wall time and
  Kaggle quota**, including setup, example and collection. The internal shared
  deadline is 25 minutes from notebook setup; the final five minutes are
  reserved for collection/shutdown. The operator must also enforce the
  external limit from actual execution start and verify GPU sessions are off.
  This proposal is not an approval; no GPU was started during local preparation.
- Keep notebook and assets Private, Internet OFF. No competition submission.

Record model load per GPU, complete-study times, overall elapsed time,
allocated/reserved memory, device maps, dtypes, cache counts/vision CUDA-event
time, logits and scores, utilization and actual Kaggle UI quota before/after.
The original two diagnostic forwards per worker are overhead, not target
forwards. Three studies do not prove hidden-test worst-case completion.
With 1,300 studies, two balanced workers and 40 s/study, the illustrative
estimate is 7 h 13 min inference, or 8 h 50 min with 20% plus ten-minute
overhead; the hidden study count and real distribution remain assumptions
to recheck before any submission.

## Local reproduction

```bash
.venv/bin/python -m pytest -q analysis/kaggle_image_t4_vision_cache_v1/tests
.venv/bin/python analysis/kaggle_image_t4_vision_cache_v1/scripts/build_notebook.py \
  --reference-csv state/kaggle_image_t4_v1/runs/20260926T151922Z/outputs/submission.csv \
  --output state/kaggle_image_t4_vision_cache_v1/private_notebook/vision_cache_diagnostic.ipynb
```

Use a new output path if the notebook already exists; the builder refuses to
overwrite a prior artifact. It embeds the checksum-bound original V8 sources,
new adapter/runner and private reference CSV. The notebook stays under ignored
`state/`; no study IDs, model weights, reports or predictions enter Git.
`source_manifest.json` records source hashes. The runner checks frozen V8
hashes in parent and child and the vision-cache source hash before loading.

After explicit approval, import the prepared notebook into the existing
private Kaggle editor, retain competition and private model assets, verify
Internet OFF / T4 x2 / Private, and start once. Download outputs and logs,
verify parity and complete-study timing evidence, then confirm no active GPU
session. Do not submit this example-only diagnostic to the competition.
