# Dataset and report-label analyses

This folder preserves dataset/report analyses and aggregate findings. Audits and postprocessing run locally; the bounded LLM smoke used a GPU. MRI training and bulk labeling of the 4,349 reports without organizer condition labels remain paused.

| Analysis | Tracked in Git | Private state |
|---|---|---|
| [Dataset label audit](label_audit/README.md) | Counts, missingness and aggregate join checks | `runs/label-audit-20260913/` |
| [Report-labeling baseline](report_labeling/README.md) | Frozen source/rules/tests, protocol, aggregate results and reproduction command | `runs/report-labeling-20260913-v1/` |
| [MedGemma / Qwen benchmark](report_labeling_llm_v1/README.md) | Pinned execution recipe, five-study smoke results, dashboard and offline error review | `runs/report-labeling-llm-v1/` inputs; `runs/report-labeling-llm-smoke-20260914/` results |
| [LLM v2 incomplete smoke](report_labeling_llm_v2/README.md) | Pinned contract, runtime, partial RTX results and diagnostic tools | `runs/report-labeling-llm-v2-ada-executed-20260915/` |
| [MedGemma v3 candidate](report_labeling_llm_v3/README.md) | Paired prompt design, shared framing, synthetic tests and CPU token sizing; NOT RUN | `runs/report-labeling-llm-v3-candidate-*/` |
| [V3 local runtime and blinded review](report_labeling_llm_v3_runtime_v1/README.md) | Bounded adapter, coded review interface, CPU software checks; GPU disabled | `runs/report-labeling-llm-v3-runtime-*/` |
| [V3 measured execution attempt](report_labeling_llm_v3_execution_v1/README.md) | Bounded attempt failed on candidate truncation; GPU stopped; 48 control technical passes, no semantic validation | `runs/report-labeling-llm-v3-execution-*/` |

Private paths are relative to `RSNA_STATE_DIR` (default: repository `state/`). They contain study identifiers, reports, splits, predictions and detailed error evidence. They are excluded from Git and included in the separate private state backup. The public GitHub repository contains code and aggregate findings only.

Current decision: **C — more validation/manual annotation before scaling report-derived supervision**. The baseline answered 92/216 held-out condition checks, with 87 correct and 5 incorrect; 124 abstained. High conditional accuracy does not establish broad coverage or reliability. Read the report-labeling [results](report_labeling/RESULTS.md) before proposing a new experiment.

The separate [candidate failure diagnostic](candidate_failure_diagnostic_v1/README.md) closes the local token/failed-child receipt audit. All nine roundtrips matched; the candidate omitted nested fields in short outputs and repeated an array in the truncated output. Historical results remain frozen; no model generation occurred.

Current user-directed focus: [Qwen saved-output audit and prospective preparation](qwen_report_extraction_v1/README.md). Both historical repeats verified; parser sensitivity does not improve the aggregate technical count. New inference is NOT RUN; MedGemma development is paused.

The [Qwen local runtime/review integration](qwen_runtime_v1/README.md) is implemented with real execution hard-disabled. Synthetic tests/demo do not represent Qwen output.

The [Qwen execution proposal](qwen_execution_v1/README.md) sets a one-hour, $1.50 ceiling for a five-report, 20-generation comparison on one RTX 6000 Ada. It includes a private approval gate, a pod-local shutdown watchdog, external stop verification, and an independent output-token decoding requirement. It is not execution approval; all RunPod rates must be rechecked before a run.

The [Qwen extraction results snapshot](qwen_extraction_review_2026-09-25_174955_sgt/README.md), dated **2026-09-25 17:49:55 Asia/Singapore**, records the completed 20-generation run's block-level totals and repeatability gate. Study-level outputs and report text remain in ignored local state.
