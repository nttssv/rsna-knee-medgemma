# MedGemma v3 — evidence-first local candidate

**Generation NOT RUN.** This candidate tests whether finding evidence before assigning a label reduces polarity, anatomy and severity mistakes. It follows the [completed offline v2 diagnostic](../report_labeling_llm_v2/response_diagnostic_v1/RESULTS.md). No new model weights, GPU startup, training or label extraction occurred during preparation.

| Item | Planned experiment |
|---|---|
| Input | Same five known development reports, original language; no images or organizer answers in prompts |
| Control | Exact v2 prompt with shared v3 response handling and output budget |
| Candidate | Evidence-first decision ladder, polarity clarification and short synthetic contrasts |
| Output | Same 12 conditions, each with label, evidence_text and confidence |
| Label states | positive / negative / uncertain / not_mentioned; never convert silence to negative |
| Shared model | Pinned MedGemma 1.5 4B, BF16, greedy; no Qwen in this candidate |
| Shared output cap | 4,096 new tokens, prospectively chosen and not yet tested in generation |
| Run order | Control 1 → Candidate 1 → Candidate 2 → Control 2 |
| Maximum design | 20 primary generations, five distinct studies |
| Current readiness | Local candidate only; no GPU adapter exists and execution is disabled |

The new prompt requires the model to find a target-specific passage, verify anatomy and qualifiers, then assign its state. It emphasizes that a relevant quotation may support a negative or uncertain answer. The software **does not correct medical labels** using keywords: an exact quotation can still be attached to the wrong diagnosis. Actual improvement must be measured and reviewed, not inferred from tests.

## Files

- [PROTOCOL.md](PROTOCOL.md): prospective design, common controls, metrics, stopping and preservation.
- [RESULTS.md](RESULTS.md): actual local checks, tokenizer counts and NOT RUN limits.
- [REVIEW.md](REVIEW.md): advisory design review and remaining work.
- [Candidate prompt](prompts/evidence_first.txt), [control prompt](prompts/control_v2.txt), [unchanged definitions](prompts/target_definitions.txt).
- [Experiment config](configs/experiment.json): exact revision, packages, decoding and disabled execution.
- [Ontology equivalence](ontology_equivalence.csv): identical definition text, with qualified review explicitly unperformed.
- [Preparation and validator](scripts/v3_candidate.py), [CPU tokenizer check](scripts/tokenize_local.py).
- [Synthetic review expectations](tests/semantic_cases.json): 24 assistant-authored cases, not measured model answers or clinical gold.
- [Historical protection manifest](configs/protected_history.json): 124 tracked analysis files preserved byte-for-byte.

## Reproduce locally

Restore private state first. From the repository root:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
python analysis/report_labeling_llm_v3/scripts/v3_candidate.py \
  --source state/runs/report-labeling-llm-v2-ada-executed-20260915 \
  --output state/runs/report-labeling-llm-v3-candidate-NEW
```

For relocated state use `--state-dir` or `RSNA_STATE_DIR`, and place source/output beneath that state root. Preparation verifies the anchored v2 manifest, fixed input digest, original unique 40/18 membership, unchanged control/definitions and historical code. It copies only the existing five-report private preparation and unresolved review queue, without reopening validation reports or evaluating validation labels. It refuses overwrites and has no inference command.

With the exact CPU tokenizer environment and ten pinned small assets already cached:

```bash
python analysis/report_labeling_llm_v3/scripts/tokenize_local.py \
  --prepared state/runs/report-labeling-llm-v3-candidate-NEW \
  --cache state/cache/tokenizer-preflight \
  --output state/runs/report-labeling-llm-v3-candidate-NEW/tokenizer-preflight
```

This checks all package versions and asset hashes before offline processor/tokenizer use. It does not instantiate a model backend, download missing files or generate answers. Rendered prompts and input IDs remain private. Keep the private state backup; GitHub contains source and aggregate findings only.

The next implementation stage, if requested, is a separately reviewed bounded adapter for this candidate. The present code cannot run the proposed paid comparison, and passing preparation/tests/tokenization does not authorize one.
