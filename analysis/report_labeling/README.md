# Frozen report-labeling baseline

The recorded experiment used **40 development / 18 validation studies** with twelve condition labels each. Code/configuration/evaluation logic was frozen before held-out extraction. See [RESULTS.md](RESULTS.md) and the original [PROTOCOL.md](PROTOCOL.md). No LLM or image predictions were used.

The `src/`, `config/`, `tests/`, protocol and recorded requirements are copied **byte-for-byte** from the frozen v1 experiment. `frozen_code_manifest.json` checks those ten files. Known mistakes are preserved so the reported validation remains reproducible. Do not fix v1 in place or tune it on the now-inspected holdout; create a separately versioned experiment with a new validation plan.

## Files and private state

```text
analysis/report_labeling/
  src/                         Frozen extraction, splitting and evaluation code
  config/                      Terminology/rules and four-state schema
  tests/                       33 synthetic semantic and metric checks
  aggregate/                   Metrics, coverage, error counts and yield projections
  PROTOCOL.md                  Design recorded before validation
  requirements.txt             Recorded environment versions
  frozen_code_manifest.json    Hashes of frozen code/config/tests
  reproduce.py                 Assemble private inputs and verify an exact replay

$RSNA_STATE_DIR/
  data/train.csv               Original organizer CSV; unchanged
  runs/report-labeling-20260913-v1/
    splits.csv                 Private study-level 40/18 assignment
    split_manifest.json        Original input and split fingerprints
    freeze_manifest.json       Original pre-validation freeze record
    evaluation_manifest.json   Original output fingerprints
    gold58_extractions.csv      Separate organizer/extracted columns and evidence
    long_extractions.csv       All 696 condition records and candidate evidence
    error_analysis.csv         Post-evaluation analyst triage
    REPORT_PORTABLE.md         Full local report with relocatable links
    TERMINOLOGY.md             Inspection notes and de-identified report examples
```

The private snapshot additionally preserves the original full outputs and development iterations. Study IDs, report excerpts, gold/extracted rows and review notes are excluded from this public repository. The tracked summaries contain aggregate measurements only.

## Reproduce locally or after migration

Use Python 3.12. The original environment was Python 3.12.13. On the current checkout, the existing `.venv` already has the needed packages. On a clean machine, install into a local virtual environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r analysis/report_labeling/requirements.txt
```

Run from the repository root after restoring private state:

```bash
export RSNA_STATE_DIR=/absolute/path/to/restored-private-state
.venv/bin/python -m pytest -q analysis/report_labeling/tests tests/test_report_analysis.py
.venv/bin/python analysis/report_labeling/reproduce.py
```

Without `RSNA_STATE_DIR`, the wrapper uses the repository's ignored `state/` folder. It creates a new timestamped run inside private state, checks the original source/split/code hashes, and compares every generated CSV to the preserved baseline. It never downloads data, starts a GPU, trains a model, changes organizer labels, or processes the 4,349 unlabeled studies. There is no bulk-labeling command.

Optional explicit paths:

```bash
.venv/bin/python analysis/report_labeling/reproduce.py \
  --state-dir /private/restored-state \
  --snapshot /private/restored-state/runs/report-labeling-20260913-v1 \
  --source /private/restored-state/data/train.csv \
  --output /private/restored-state/runs/report-labeling-reproduction-check
```

The output directory must be new and inside the chosen state directory. This command is a replay of the frozen result, **not another independent validation**. It needs the original private split; cloning GitHub alone is insufficient. The existing state bundle includes both analysis snapshots because they are under `state/runs/`; see [migration instructions](../../docs/MIGRATION.md).

## Interpretation

States are positive, negative, uncertain and not_mentioned. The last means **no supported rule match**, which may reflect an unsupported language or missed terminology. It is never automatically negative. Confidence is an uncalibrated rule-strength score, not a probability of disease.

TP/FP/TN/FN and the usual binary metrics use the decided subset. Always report its denominator, coverage and abstentions. Macro metrics show the number of conditions with defined values. Three Bulgarian holdout reports were unsupported, and several conditions had only one output sign or a single positive example. No condition met the conservative predeclared scaling gate.

For future work, independently validate condition/language-specific rules and report/reference disagreements. A separate LLM method may be compared with the baseline, but no LLM extraction has been implemented or evaluated in v1.
