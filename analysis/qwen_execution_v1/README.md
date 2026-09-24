# Qwen GPU execution proposal

**Status: source review complete; no critical blockers identified. GPU execution remains disabled.** This package defines the next Qwen step as a concrete, cost-bounded proposal without starting paid compute.

| Boundary | Proposal |
|---|---|
| Work | Five fixed development reports, two prompt arms × two repeats |
| Maximum generations | 20 |
| Model | `Qwen/Qwen3-14B` at revision `40c069824f4251a91eefaf281ebe4c544efd3e18` |
| GPU | One RTX 6000 Ada, 48 GiB |
| Provider window | At most 60 minutes, including setup and transfer |
| Inference window | At most 30 minutes and must finish before watchdog stop time |
| Budget | At most $1.50; current Secure Cloud compute ceiling $0.84/hour |
| Pod storage | Secure Cloud, 80 GB container disk, no persistent or network volume |
| Shutdown | Watchdog stops at T−5 min regardless of model state; hard provider deadline at T; external state verification |
| Current execution | Disabled; no approval record exists |

The [protocol](PROTOCOL.md) explains measured sizing evidence, required CUDA/cache checks, independent output-token decoding and failure rules. The private grant binds both the hard provider deadline and the watchdog stop time, plus Secure Cloud, exact container disk size, zero persistent volume and no network volume. Inference must complete before the stop time; the watchdog will stop the pod even if a model process is still running. [Resource settings](configs/resource_proposal.json) expire after 24 hours and must be refreshed against the signed-in console before any start. The CPU test suite exercises bad/expired grants, cost and deadline limits, exact decoding, pod targeting and bounded stop attempts without contacting RunPod.

Refreshed public quote (2026-09-24 12:49 UTC): RunPod lists $0.74/hour Community Cloud and $0.84/hour Secure Cloud for this GPU; this proposal uses Secure. Container disk is listed at $0.10/GB/month, so 80 GB is about $8/month or $0.011/hour. The rounded quote is about $0.851 for 60 minutes; the proposal gate allows up to $0.852/hour-window total ($0.84 compute + $0.012 storage allowance), within the separate $1.50 cap. The public quote is valid through 2026-09-25 12:49 UTC. Compare with the signed-in deployment quote before any start. The signed-in console currently shows only stopped Pods at $0.00/hour, not a configured new allocation quote. This estimate is not a provider-enforced dollar cap.

These are observed prices, not authorization. A run needs fresh explicit approval tied to the exact plan, proposal, GPU, cloud/storage configuration, pod, displayed rate, cost cap and both deadlines. The watchdog begins stop attempts five minutes before the hard provider boundary, leaving that reserved interval for retries and external verification. The prior temporary account key was revoked and cannot be reused; any broader replacement credential needs separate approval. Before a start, verify the same control path can read and accept stop for the exact target while it is already stopped, then verify its stopped state again and retain a private receipt. Earlier SSH omitted `RUNPOD_POD_ID`, and the built-in pod credential failed its read check. The prior live watchdog was armed, but the operator stopped early, so its scheduled stop was never exercised. The new runtime stage must verify its environment and access against the preflight receipt before loading the model. See the [recorded access findings](../report_labeling_llm_v3_execution_v1/SETUP_ATTEMPT.md) and [stopped-pod preflight](../report_labeling_llm_v3_execution_v1/CONTROL_PREFLIGHT.md).

The next action, after source review, is a user decision on this exact resource boundary. Approval would authorize a separately recorded one-time execution; it would not authorize training, validation, bulk extraction, retries, another GPU or a budget extension.
