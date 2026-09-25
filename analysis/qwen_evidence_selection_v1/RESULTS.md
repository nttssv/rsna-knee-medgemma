# Results

**REAL QWEN NOT RUN.** The A/B package and supervised runner are implemented; no new Qwen model generation, training, validation inference, GPU execution or provider mutation has occurred.

The offline dry run passed exact A/B prompt and input-token parity against the private five-report package. The supervised CPU rehearsal called a fake backend exactly 20 times in A1 → B1 → B2 → A2 order, five reports/block, and wrote 20 durable attempt/raw/parsed records with `SYNTHETIC: true`, zero unrun and zero GPU memory. Separate synthetic checks rejected a repeated output path, altered prompt or token artifact, absent real-session grant and incomplete generation (one attempted, zero completed, one failed, 19 unrun). A hard-timeout rehearsal killed the worker process group and retained partial artifact hashes; parser-invalid synthetic responses were recorded as technical failures without retry or repair. These are adapter checks, not report-extraction or medical-accuracy results.

Source review at commit `2627428d443edf16803dfe18c2f68b0d356a2141` identified two finalization/recipe bindings. The follow-up CPU tests inject failure while writing the final parsed record and final A2 block receipt; both now return failed/nonzero despite 20 completed generations. The adapter also binds the historical Qwen config actually used by `real_encoder()` and verifies the effective encoder recipe against the A/B plan before loading weights. Seed, output-cap and revision drift fail closed. Public CI runs source checks; private five-report rehearsal remains local and is skipped when those private inputs are absent.

New execution-plan SHA-256: `6d551f7bf19db648e11eed9766e6a47b984567e6b8fbc6e4dd660c738bf9c54a`. Private reports, study IDs, raw outputs and the regression checklist remain outside Git.

## Frozen baseline used to design B

The public aggregate snapshot at commit `2f6604bb68348dd275554e66f539869763a1bff0` is the numerical reference. In control-1 vs candidate-1, 20/60 raw proposals differ; 18/60 primary parsed rows differ. The remaining two cells differ only in raw proposals and collapse to the same primary parsed row.

The private local report review and regression checklist identify two distinct Contusion checks: reject muscle-only contusion as evidence for the bone-marrow target, and retain an explicit negative statement about bone contusion rather than leaving it `not_mentioned`. These are provisional review flags, not medical adjudications. No historical output has been modified or marked resolved, and organizer labels were not opened.

A/B output comparison, repeatability, technical and semantic validity, and the single experiment recommendation remain pending a separately approved run.
