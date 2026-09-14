# V2 results

**STATUS: NOT RUN.** No v2 LLM outputs, validation metrics, agreement scores, measured GPU time or measured VRAM exist.

The local implementation provides strict JSON/outer-fence handling, narrowly defined whitespace matching with original source offsets, no generated primary repair, label-preserving technical failures, unchanged model revisions, paired prompt instructions, a fail-closed engineering review gate and private five-development-study preparation.

Local verification passed **167 tests in the full repository suite**, including 83 v2 checks. Twenty-nine v2 checks serialize reviewer-authored ontology/specification expectations: they are **not** tests that an LLM predicts those labels correctly. All 70 protected tracked v1 files remain unchanged. Private preparation produced exactly five historical development inputs and 12 unresolved review candidates, with zero model calls. The same five input IDs/order were verified against all four original model runs. The validation input file was not opened by preparation; split metadata is used only to preserve original membership.

The queue comprises nine binary organizer discordances, two v1 evidence failures and one questionable uncertain output. No reference was corrected and no model answer was adjudicated. [Public counts](aggregate/adjudication_counts.csv) describe that inherited review workload, not v2 predictions.

No v1 output was reprocessed under the new acceptance rules to manufacture v2 scores. A future smoke must generate new model outputs under a reviewed, pinned v2 execution recipe. Both historical result directories and their metrics remain intact.

The private review viewer and ten-panel dashboard were generated with explicit NOT RUN placeholders. They show the selected report inputs, organizer references and frozen rule outputs; they contain no v2 prediction or measured model metric. Future artifact loading verifies provenance and raw-response parsing before rendering results.

Next decision: resolve the [runtime questions](REVIEW.md), then consider the [bounded resource proposal](RESOURCES.md). Keep full development, validation, MRI training and bulk weak-label extraction paused.
