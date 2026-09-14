# Dataset and report-label analyses

This folder preserves local CPU analyses and their aggregate findings. No GPU is needed. MRI training and bulk labeling of the 4,349 reports without organizer condition labels remain paused.

| Analysis | Tracked in Git | Private state |
|---|---|---|
| [Dataset label audit](label_audit/README.md) | Counts, missingness and aggregate join checks | `runs/label-audit-20260913/` |
| [Report-labeling baseline](report_labeling/README.md) | Frozen source/rules/tests, protocol, aggregate results and reproduction command | `runs/report-labeling-20260913-v1/` |
| [MedGemma / Qwen benchmark](report_labeling_llm_v1/README.md) | New implementation, draft prompts/configs, protocol, resource proposal and NOT RUN status | `runs/report-labeling-llm-v1/` |

Private paths are relative to `RSNA_STATE_DIR` (default: repository `state/`). They contain study identifiers, reports, splits, predictions and detailed error evidence. They are excluded from Git and included in the separate private state backup. The public GitHub repository contains code and aggregate findings only.

Current decision: **C — more validation/manual annotation before scaling report-derived supervision**. The baseline answered 92/216 held-out condition checks, with 87 correct and 5 incorrect; 124 abstained. High conditional accuracy does not establish broad coverage or reliability. Read the report-labeling [results](report_labeling/RESULTS.md) before proposing a new experiment.
