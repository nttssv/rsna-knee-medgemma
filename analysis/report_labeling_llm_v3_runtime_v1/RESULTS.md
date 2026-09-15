# Local integration results

**No MedGemma inference was performed. GPU execution remains disabled.** This package implements runtime/review software for the frozen v3 comparison, not a new clinical result.

The CPU demo fabricates five reports and twenty fixed JSON responses across four subprocesses in ABBA order. Those responses deliberately say `not_mentioned` for every condition. Their 240 technical passes test accounting and cannot be interpreted as MedGemma accuracy, semantic improvement or usable labels. Semantic review remains blank in the user-facing demo; tests fill separate synthetic review fixtures solely to exercise validation and stage-two export.

The implementation records exact inputs/outputs, checks session provenance and artifact hashes, reparses raw responses, rejects incomplete sessions and creates a coded review interface. Organizer references and the arm mapping are excluded from the first-stage reviewer directory. Stage-two numeric comparison is unlocked only by complete validated first-stage review; clinical reviewer qualifications are not certified by the tool.

All original baseline/v1/v2/v3 analysis files, prior private outputs and the exact 40-development/18-validation split are preserved. The five real reports remain four English and one Spanish; no new translation or validation inference was performed. The inherited 12 adjudication items are still unresolved.

The real CUDA branch has not been run or benchmarked. GPU memory at 4096 output tokens, end-to-end duration, completion yield and semantic quality are unknown. The 1800-second worker and 3600-second session limits are enforced safety bounds, not time forecasts. A future run needs a reviewed execution revision and a current quote/resource authorization. No MRI training, YOLO work, full development/validation inference or bulk labeling is enabled.

See [PROTOCOL.md](PROTOCOL.md) for the tested contract and remaining limits. Detailed local verification counts and source review are recorded in [verification.json](aggregate/verification.json) and [REVIEW.md](REVIEW.md).
