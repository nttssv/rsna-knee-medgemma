# MedGemma / Qwen report-extraction benchmark

**Status: implemented locally; real LLM inference NOT RUN.** Model downloads, tokenizer preflight, development inference, prompt finalization, repeatability and validation remain pending execution approval. See [RESULTS.md](RESULTS.md), [PROTOCOL.md](PROTOCOL.md) and the [resource proposal](RESOURCES.md).

The question is whether text-only LLM extraction can improve **correct binary-label yield across all condition checks** while retaining reliability. Compare pinned MedGemma 1.5 4B and Qwen3 14B with the existing frozen rule baseline. This is a practical, unequal-model-size comparison, not an experiment isolating medical specialization.

Every file under `analysis/report_labeling/` remains unchanged. The original 40 development / 18 validation assignment is reused exactly. Because the validation outcomes were inspected previously, new results will be exploratory and cannot establish readiness to label the remaining 4,349 reports. No MRI training, image processing, pilot-adapter loading or bulk-report labeling occurs here.

## Contents

```text
README.md / PROTOCOL.md / RESULTS.md    Method, execution status and evidence limits
RESOURCES.md / REVIEW.md               Exact model proposal and external design review
configs/                              JSON-compatible YAML; no secrets or provider paths
prompts/                              Draft common instructions and target definitions
scripts/benchmark_core.py              Fingerprints, label-free inputs, output validation
scripts/model_runtime.py               Optional Transformers text-only backend
scripts/prepare_benchmark.py           Restrict input to the existing 58-study split
scripts/hardware_audit.py              Read-only machine/package/cache audit
scripts/tokenizer_preflight.py         Check both full rendered prompts on all 58
scripts/run_medgemma.py                Dry-run by default; explicit execution required
scripts/run_qwen.py                    Dry-run by default; explicit execution required
scripts/repeatability_check.py         Compare distinct five-study development runs
scripts/freeze_benchmark.py            Require completed development and repeatability
scripts/evaluate_llm.py                Verified primary and secondary comparisons
scripts/agreement_analysis.py          Same unified evaluation entry point
scripts/language_analysis.py           Same unified evaluation entry point
scripts/analysis_metrics.py            Abstention accounting and paired bootstrap
aggregate/                            Verified baseline reference + execution status only
```

Per-study state belongs in `$RSNA_STATE_DIR/runs/report-labeling-llm-v1/`. The current local preparation contains `inputs/development.jsonl` and `inputs/validation.jsonl` plus their manifest, with no organizer label fields. Source metadata and original split/language artifacts must be restored from the private migration backup. Cloning the public repository alone is sufficient for synthetic tests, but insufficient for reproducing the real study analysis.

## Public-safe CPU verification

Use the existing repository CPU environment, or install it according to the root README. No Transformers package, tokenizer, model weights, reports or GPU are needed for these tests:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_llm_benchmark.py
PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/python analysis/report_labeling_llm_v1/scripts/hardware_audit.py
```

The explicit `PYTHONPATH=src` also works in local environments where editable-install path hooks are not loaded. Tests use synthetic identifiers, synthetic text and injected generator responses; passing them does not establish actual model loading or CUDA correctness.

## Prepare private inputs and dry-run

Run from the repository root. `RSNA_STATE_DIR` defaults to ignored `state/`; an absolute external private directory also works. These commands do not download or run models. Preparation refuses to overwrite existing inputs; reuse the existing prepared folder if it is already present.

```bash
export RSNA_STATE_DIR=/absolute/path/to/private-state
LLM_RUNS="$RSNA_STATE_DIR/runs/report-labeling-llm-v1"
LLM_SCRIPTS=analysis/report_labeling_llm_v1/scripts
python "$LLM_SCRIPTS/prepare_benchmark.py" --output "$LLM_RUNS/inputs"
python "$LLM_SCRIPTS/run_medgemma.py" --prepared "$LLM_RUNS/inputs" \
  --output "$LLM_RUNS/medgemma-smoke1" --limit 5
python "$LLM_SCRIPTS/run_qwen.py" --prepared "$LLM_RUNS/inputs" \
  --output "$LLM_RUNS/qwen-smoke1" --limit 5
```

Only the report string enters the model prompt. Join IDs and organizer labels are excluded. Both models use identical extraction instructions and their own official chat template. Technical failures remain separate from the four medical states. A strict exact-source quotation is necessary for a binary decision, but its presence does not prove semantic correctness.

## Execution sequence after user resource approval

These steps are documented for the next approved stage, **not executed in this release**. Use a separate Linux/CUDA environment; do not upgrade the saved MRI pilot environment in place:

```bash
python3.12 -m venv /private-environments/rsna-report-llm
source /private-environments/rsna-report-llm/bin/activate
python -m pip install -r analysis/report_labeling_llm_v1/requirements-gpu.txt
python "$LLM_SCRIPTS/hardware_audit.py" --output "$LLM_RUNS/hardware.json"
python "$LLM_SCRIPTS/tokenizer_preflight.py" --prepared "$LLM_RUNS/inputs" \
  --output "$LLM_RUNS/tokenizer-preflight" --execute --allow-download
```

`--allow-download` enables normal Hugging Face file acquisition using existing access. Without it, loaders require cached files. MedGemma requires the user's previously granted gated access. No script accepts model terms or starts/stops a provider instance. A stopped GPU cannot provide a live CUDA/package/free-disk audit; perform that before model loading.

Run five development studies twice for each model using distinct output directories. For example:

```bash
python "$LLM_SCRIPTS/run_medgemma.py" --prepared "$LLM_RUNS/inputs" \
  --tokenizer-preflight "$LLM_RUNS/tokenizer-preflight" \
  --output "$LLM_RUNS/medgemma-smoke1" --limit 5 --execute --allow-download
python "$LLM_SCRIPTS/run_medgemma.py" --prepared "$LLM_RUNS/inputs" \
  --tokenizer-preflight "$LLM_RUNS/tokenizer-preflight" \
  --output "$LLM_RUNS/medgemma-smoke2" --limit 5 --execute
```

Repeat with `run_qwen.py` and Qwen output names. Check repeatability with `repeatability_check.py --prepared ... --medgemma-first ... --medgemma-repeat ... --qwen-first ... --qwen-repeat ... --output "$LLM_RUNS/repeatability.json"`. The check rejects repeated technical failures. Review smoke outputs before allocating a larger inference window.

When the development recipe is finalized, run each model on all 40 development studies by omitting `--limit`. Then freeze and run the complete validation partition:

```bash
python "$LLM_SCRIPTS/freeze_benchmark.py" --prepared "$LLM_RUNS/inputs" \
  --medgemma-run "$LLM_RUNS/medgemma-development" --qwen-run "$LLM_RUNS/qwen-development" \
  --repeatability "$LLM_RUNS/repeatability.json" --output "$LLM_RUNS/freeze.json"
python "$LLM_SCRIPTS/run_medgemma.py" --prepared "$LLM_RUNS/inputs" \
  --tokenizer-preflight "$LLM_RUNS/tokenizer-preflight" --partition validation \
  --freeze "$LLM_RUNS/freeze.json" --output "$LLM_RUNS/medgemma-validation" --execute
python "$LLM_SCRIPTS/run_qwen.py" --prepared "$LLM_RUNS/inputs" \
  --tokenizer-preflight "$LLM_RUNS/tokenizer-preflight" --partition validation \
  --freeze "$LLM_RUNS/freeze.json" --output "$LLM_RUNS/qwen-validation" --execute
python "$LLM_SCRIPTS/evaluate_llm.py" --prepared "$LLM_RUNS/inputs" \
  --medgemma-run "$LLM_RUNS/medgemma-validation" --qwen-run "$LLM_RUNS/qwen-validation" \
  --freeze "$LLM_RUNS/freeze.json" --output "$LLM_RUNS/evaluation"
```

Development evaluation uses `--partition development` and does not require a validation freeze. Any change to scripts, prompts, configs, protocol, requirements or benchmark tests invalidates earlier code hashes; regenerate tokenizer preflight and repeat affected development runs before freezing. Do not modify a frozen version after inspecting validation outcomes.

## Expected real-run outputs

Each private run retains exact rendered prompts, raw responses, first-pass and second-pass records, evidence offsets, failure statuses, token counts, runtime and provenance. A startup failure leaves `started_manifest.json`; only a complete run has `run_manifest.json`. Infrastructure errors have no automatic retry. Partial runs cannot masquerade as the full 40/18 benchmark.

Evaluation writes separate `first_pass/` and `repair_assisted/` folders containing per-model metrics, model comparison, binary-agreement metrics, descriptive four-state agreement, language metrics, technical-failure counts, overall study-bootstrap intervals, paired deltas and condition recommendations. Detailed `private_disagreements.csv` remains private. Its error categories begin as unreviewed; diagnostic failure attribution requires manual review and is not fabricated automatically.

After real execution, review and copy only the intended aggregate CSVs into `aggregate/`, update `RESULTS.md` with measured values, verify no study-level material or credentials are staged, and commit. Do not publish empty scores as measured results or call the first-pass and second-pass results independent experiments. Self-reported confidence and model agreement are not ground truth.
