# Protocol

## Question and design

Does adding a concise whole-report evidence-selection workflow to the successful control prompt improve report-supported evidence selection on the same five development reports? This is a paired repeated-measures prompt comparison, not an independent sample of ten reports. A1 → B1 → B2 → A2; five reports per block; at most 20 generations. The public snapshot at commit `2f6604bb68348dd275554e66f539869763a1bff0` is the aggregate baseline; its representative control-1/candidate-1 comparison has 20/60 raw proposals differing, while 18/60 primary parsed rows differ (two differences occur only in raw proposals).

A is frozen byte-for-byte at SHA-256 `86b3fdf3adffff5e74b3b94192580c5d771711d441d682ecba88cc64bb344a68`. B is exactly A with one evidence-selection instruction inserted before the existing final output-only instruction. Both use the same target definitions, one generation per report, Qwen3-14B at the pinned revision, non-thinking BF16, SDPA, greedy batch 1, seed 20260914, input cap 8192, output cap 2048, context cap 40960, and stop IDs `[151645,151643]`. Primary v2 and secondary v1 parsers remain unchanged.

The prompt asks the model to silently review Findings and Impression together, match evidence by target anatomy/tissue/compartment/time, apply a statement to each explicitly named target, reconcile same-target negation and qualifiers, then select one continuous exact span. It does not include any regression case, study identifier, original report phrase, organizer label, reasoning trace, extra model call or additional target definition.

## Frozen data and protections

The source is the same five original-language development reports bound by prepared plan SHA-256 `c4586bb0c65163244dac80344ea96f691493a1b89a5911eefcba54f2384c74fc`. Input SHA-256 is `0a4b05ccca46ff85fd1befc077155b40837dd4787603a91b579500f04cbe5ce8`; split SHA-256 is `0900a46a7d47c47451ce14c4f18237b31b72cd90b19d3af535d744ad07ef1133`. The organizer labels are not used in prompt construction or first-pass content review. Existing prompts, parsers, reports and outputs remain unchanged.

The default entry point remains dry-run and execution is not approved. A future real run must explicitly use `--run` and a fresh private user grant bound to the new execution-plan SHA, prompt-bound prepared package, exact pod/resource/rate, session budget and immutable deadlines. A current independently obtained RUNNING observation and live pod-local stop receipt are mandatory. The worker checks that process through the same pod's `/proc` before every generation. The reused process-group supervisor kills and reaps a stuck worker before 1,800 seconds, while the separate exact-pod stop process is a billing backstop. The operator must also retain a tested outside-pod stop route. The launcher reserves at least 15 minutes for copy/stop, and the stop backstop starts at least five minutes before the hard provider deadline. No prompt changes, retries, repairs, repeat selection, validation inference, full-40 inference, bulk extraction, MRI training or fine-tuning are allowed.

## Local checks and token budget

Use only the pinned Qwen tokenizer snapshot already present locally, with `local_files_only=True`. Require exact A rendered-input/token-ID parity with the prior prepared artifacts. B must fit the existing 8192 input and 40960 total-context limits with the same 2048 output allowance. Missing cache, version/template mismatch or any input-integrity mismatch blocks preparation; do not download model weights or increase caps.

The private regression checklist includes both observed Contusion error types: treating muscle-only contusion as the bone-marrow target, and missing an explicit target-matched bone-contusion negative. These are review targets, not resolved adjudications. The 12 synthetic contrastive fixtures are development specifications only. They do not enter any model prompt or count toward the 20 calls. CPU tests and the supervised fake-backend rehearsal verify the actual A/B dispatch, durable records, frozen parsing/independent decoding, source/token integrity and timeout/failure behavior; they cannot establish that Qwen reasons correctly. All fake output is marked `SYNTHETIC`.

## Evaluation after approval

Report technical validity separately from evidence adequacy, whole-report consistency, explicit-evidence omission, semantic abstention and repeatability. Summarize all 60 condition rows for each block. Preserve raw-proposal differences separately from accepted parsed rows. Include every technical failure and all repeated-prompt raw responses. The historical repeatability gate remains 60/60 valid-and-identical; matching technical failures still fail. Do not treat confidence or increased abstention as quality. Only human-reviewed cells count as semantically reviewed; all other cells are `NOT ADJUDICATED`. The private regression checklist is for post-run review only and cannot be used to patch outputs.
