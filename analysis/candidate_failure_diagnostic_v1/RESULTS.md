# Why the candidate failed again

**No prompt-rendering, token-decoding or saved-receipt mismatch was found. The candidate produced two different forms of output-contract failure.** This is a local audit of existing evidence, with zero model loads or new generations. It does not turn the failed experiment into a successful comparison.

| Check | Observed result |
|---|---|
| Rendered prompts and input token sequences | Exact match for all 9 recorded calls |
| Full decoded output and raw output with original EOS handling | Exact match for all 9 calls |
| Preflight input counts | All 10 reconstructed, including the unrun fifth candidate input |
| Parent and child receipts | Completed control and failed candidate bindings verified |
| Cached tokenizer/processor assets | Matched pinned hashes; offline only |
| Original full model-cache audit | All 13 recorded file entries matched the pinned manifest; weights not reloaded |
| Historical analysis files | All 181 byte-identical |

## Failure 1: short answers discarded required fields

The first three candidate responses contained exactly the 12 condition keys, but **all 36 values were strings containing state names**. The required values were nested objects containing a label, evidence and confidence. Their combined 406 output tokens reflect abbreviated output, not a token-budget shortage. They had no recorded thought envelope and remained 36 schema failures. No string was converted into an accepted label.

This establishes the observed shape violation. It does not establish that the state names were medically correct. The hypothesis that the longer evidence-first instruction encouraged abbreviated answers is plausible but unproven; there was no completed repeat comparison.

## Failure 2: the long answer repeated a list

The fourth candidate response began a Markdown-fenced JSON **array**, rather than the required object keyed by the 12 conditions. It contained **95 object-shaped lines but only 11 distinct exact lines**; the most frequent exact line appeared 11 times. These were repeated evidence objects, not one bounded entry per target condition.

All **4,096 tokens were in the recorded answer surface**: no `<unused94>` opening marker, no `<unused95>` closing marker, and no terminal EOS. An identical 16-token window appeared 94 times; repeated-window excess was 3,686 among 4,081 overlapping windows. Overlapping windows are correlated, so these are descriptive repetition measures, not independent events or an estimated probability of looping.

The official status remains `generation_truncated`. Its 211.55-second generation hit the token cap before the 300-second time limit. The fifth candidate report and both second repeats were not dispatched. Merely raising the token limit would not correct the already-wrong array shape; whether it would eventually terminate is unknown.

**Correction to the earlier explanation:** it was reasonable to suspect long reasoning from the previous v2 failure, but this v3 candidate truncation occurred in repetitive answer text, with no recorded thought envelope. This does not reveal unobservable internal reasoning; it describes the saved token stream.

## What the control does and does not tell us

Across five control responses, 7,883 tokens were inside recorded thought envelopes and 2,764 were answer text; ten delimiter tokens plus five terminal EOS tokens account for the remaining 15 tokens. All completed, but only 48/60 condition cells passed the frozen structural/evidence checks. Thus a thought envelope alone does not explain the candidate failure. Those 48 technical passes still lack qualified semantic review.

The same pinned tokenizer reconstructed the exact inputs/outputs for both arms. This removes a detected serialization/decoding discrepancy as an explanation for these saved responses. It does not prove all GPU computation was correct or isolate why the model chose a different output shape. The five inspected development studies and missing repeats cannot establish generalization or a winning prompt.

## Next decision

Keep the v3 failure frozen. **The next proposed correction should target a fixed, condition-keyed output contract**, with the same explicit schema placement across arms, rather than immediately buying more compute or increasing the cap. Any constrained-output implementation must first be verified against the pinned model/runtime and synthetic cases; support and clinical benefit have not been demonstrated here. A prompt-only change may still fail and must not be called enforcement.

Treat the short missing-field responses and the long repeated array as separate failure modes. Do not repair this run, promote its string values to labels, choose favorable outputs, or resume its unfinished comparison. A prospective redesign and qualified semantic review remain separate work. No paid run, full-development/validation inference, bulk extraction, MRI training or YOLO work is authorized by this audit.

[Aggregate measurements](aggregate/summary.json) · [Audit protocol](PROTOCOL.md) · [Reproduction and checks](README.md)
