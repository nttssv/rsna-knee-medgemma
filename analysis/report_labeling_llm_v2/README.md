# LLM report extraction v2 — incomplete smoke

**INFERENCE STATUS: INCOMPLETE / FAILED SMOKE.** The user-authorized RTX 6000 Ada run attempted three reports on 2026-09-15, then stopped on truncation. None of the 36 attempted condition cells was accepted. MedGemma repeat 2 and both Qwen repeats were NOT RUN. The GPU was stopped after all 17 original artifacts were copied and hash-verified. Read [measured RTX results](RTX_SMOKE_RESULTS.md) and use the separate [partial diagnostic viewer](diagnostics/README.md). No MRI training, validation inference or bulk labeling occurred. A subsequent [offline response diagnostic](response_diagnostic_v1/RESULTS.md) found framing, truncation and semantic problems; prefix removal alone is insufficient.

The aim is to address general format, evidence and semantic failure modes from the [v1 smoke](../report_labeling_llm_v1/RESULTS.md). MedGemma remains the intended teacher candidate and Qwen the comparator; both face the same quality gate. A report extractor is not yet a trained MRI classifier or a source of validated training labels.

## What changes

| Aspect | Preserved v1 | Prospective v2 |
|---|---|---|
| JSON acceptance | Bare strict JSON | Bare JSON or exactly one whole outer `json`/untyped fence |
| Evidence | Exact substring | Unique exact occurrence first; space/tab/CR/LF fallback only if absent |
| Primary output | First generation; repair separately recorded | One generation; deterministic normalization only; no repair generation planned |
| Semantic instructions | Original common prompt | General anatomy, scope, severity, polarity and timing clarifications |
| Predictions | Existing measured smoke | Incomplete three-report smoke; no v2 accuracy claims |

Both v1 directories, all saved v1 outputs and the original 40/18 split remain unchanged. [Protected hashes](configs/protected_v1.json) anchor 70 tracked files at commit `c33e50da96046b9ebd4c845361802089cfcd2483`. No v1 output has been re-scored under the v2 policy.

## Local use

From the repository root, run the complete suite:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

Prepare a **new** private directory after restoring the original private state:

```bash
python analysis/report_labeling_llm_v2/scripts/prepare_local.py \
  --output state/runs/report-labeling-llm-v2-local-NEW
```

Use `--state-dir` or `RSNA_STATE_DIR` for an external private persistent disk, and keep `--output` underneath it. Preparation verifies source/split and all four original prediction-file hashes. It reads the development input only, selects the same first five reports and verifies their order against all four v1 runs. It does not open validation report inputs or evaluate validation labels. The original split file is read for membership only; the source CSV is hashed and streamed, with diagnostic columns evaluated only for these five development IDs.

Outputs: five label-free input records, two sets of unrendered model prompts, a private unresolved adjudication queue, organizer/rule review context, and a provenance manifest. The prompt strings contain reports and instructions; study IDs and organizer labels are excluded. These are not model outputs or tokenizer preflight results. Preparation rejects an existing output directory.

## Files

- [PROTOCOL.md](PROTOCOL.md): exact normalization, states, repeatability and advancement criteria.
- [LOAD_PREFLIGHT.md](LOAD_PREFLIGHT.md): separate full-cache and bounded GPU load check; no inference.
- [RUNTIME.md](RUNTIME.md): adapter, fake-backend tests, template/stop-token review and locked execution command.
- [RESULTS.md](RESULTS.md): CPU verification and measured incomplete smoke status.
- [REVIEW.md](REVIEW.md): external advisory context and unresolved questions.
- [RESOURCES.md](RESOURCES.md): historical measurements and proposed bounded smoke.
- [ADJUDICATION.md](ADJUDICATION.md): qualified-review workflow; no automatic reference correction.
- [core.py](scripts/core.py): normalization, exact source offsets and strict schema validation.
- [gates.py](scripts/gates.py): software checks plus a mandatory human review with per-cell provenance; never authorizes compute.
- [Review viewer and dashboard](smoke_review_tools/README.md): historical preview and completed-session gate; partial diagnostics remain separate.
- [synthetic review cases](tests/semantic_cases.json): intended protocol behavior for human/model review, not measured model answers.

The checker verifies syntax and lexical grounding, **not medical entailment**. A wrong label attached to an exact quotation may pass its technical checks. Every accepted row retains `semantic_review_required=true`; synthetic fixture tests do not prove that either LLM follows the instructions. The [local inference adapter](RUNTIME.md) has synthetic generation tests and real model load-only checks. CPU tokenization passed for both pinned snapshots; the historical MedGemma export comparison is preserved separately. Full weight-cache and CUDA/BF16 load checks also passed. MedGemma generation was measured on three reports; the output contract failed. Qwen generation and repeat consistency remain unmeasured for v2.
