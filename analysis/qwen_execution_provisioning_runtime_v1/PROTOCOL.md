# Protocol: local outer lifecycle controller

This package implements and tests a local state machine. It does not implement a safe live RunPod client. Its CLI refuses live execution unconditionally. Supplied approval files in tests are synthetic fixtures, not user approvals.

## Source and experimental bindings

- Provisioning proposal SHA-256: `0770846122bb4bb4953c7e728afa1c254ae569c8f756b3e905e6c095a2ed2842`.
- Reviewed inner execution plan: `ab82f51cf3883e93b2c33bf750a32148297226ebd3e9ada0febb54d18d85a4f5`.
- Prepared plan: `c4586bb0c65163244dac80344ea96f691493a1b89a5911eefcba54f2384c74fc`.
- Alternate resource proposal: `884dc38d2004abb44c64d2363e525969bf7f7fabd35f86eb5d03144bf34e72e2`.

The new outer approval design additionally binds the exact controller source SHA, immutable image reference, exact startup command hash, high-entropy unique creation intent, fixed A40/region/storage specification, actual compute/storage rates, maximum budget, provider-window length, approval time, and fresh post-approval selected-allocation quote. The file must already exist as an owner-only 0600 regular file. Symlinks and duplicate JSON keys fail. The code never creates a user approval.

Qwen/Qwen3-14B, pinned revision `40c069824f4251a91eefaf281ebe4c544efd3e18`, BF16, SDPA, batch size 1, both prompts/parsers, input/output/context limits, EOS IDs, five reports, ABBA20 sequence, and 40/18 split remain in the unchanged inner plan. This controller has no path for loading that model or launching the inner inference worker.

## Before the sole create request

1. Validate the supplied outer approval, frozen source hashes, exact allocation and quote. At least one positive compute rate and nonnegative finite storage rate must fit their respective ceilings. The approved outer budget must conservatively cover the full allowed wall-clock window at the $0.502/hour ceiling.
2. Create a new private output directory exclusively. Existing output directories cannot be reused to retry or recover a session.
3. Read the provider by one exact `qwen-provision-` name with 128 random bits. Any existing match ends preparation without creating or stopping anything.
4. Durably write an immutable intent containing the first-create timestamp, deadline no later than 3,600 seconds afterward, stop time exactly deadline minus 300 seconds, allocation, approval/source hashes and cumulative ceiling. The timestamp is recorded just before guard verification; that verification delay is conservatively inside the outer window.
5. Verify an independent external worker against that exact intent hash/name and immutable deadlines. Its process must be separate from the allocating process, alive, stop-capable, and independently verified for command/environment/receipt identity. A parsed PID receipt alone cannot establish this; the injected `ExternalGuard.verify` contract requires an independently implemented verifier. **That live verifier/worker does not ship here.**
6. Recheck immutable intent/authorization bytes, current quote and time window. Fsync the exclusive create-attempt receipt before dispatch. A failure to write that receipt prevents the create call.

## Create, reconcile, stop

Send one request for the exact image, sleep-only CMD, region/GPU/storage specification. No mutable template is selected. A returned pod ID is bound immediately for cleanup. Read the provider's current allocation and physical machine ID. Every resource field, actual rate, image reference and command must match before accepting the allocation as valid. Unknown or unavailable fields fail; they are not inferred from the requested draft.

If the response is lost, issue one read-only exact-intent-name reconciliation. No match or multiple matches leave identity unresolved: no guessed ID is stopped and no creation retry occurs. The external worker/operator must resolve and safely shut down any billable resource. For exactly one match with the preread-absent high-entropy name and valid pod ID, bind that ID for cleanup before checking configuration. A configuration mismatch still stops that uniquely identified resource, but never accepts it for later use.

Once a pod ID is known, cleanup executes on success and failure. Up to three stop requests are allowed, each with a requested timeout no longer than 30 seconds or the remaining outer window. Read stopped state and zero hourly charge through the independent observer after each attempt. Reads must be fresh within five seconds and name the exact pod. A lost stop response may still be confirmed by the independent read. If local logging fails, emergency stop/read calls continue; lost durability blocks all subsequent progress. No stop response is treated as billing confirmation.

The current local adapters are synchronous. A timeout argument is an adapter contract, **not a hard bound on a stuck network call**. The future production controller must supervise each call in a killable/reaped child and keep the independently deployed external worker alive through allocating-process death. Local synthetic tests do not establish this live guarantee.

## Cumulative accounting and later grant

For a verified allocation, accrue an upper bound at $0.502/hour from first-create timestamp to independently verified stop, including ambiguous-response time and all control latency. Compute with Decimal and round upward to one nano-dollar. If allocation/rate readback fails or is unknown, record the cost bound as `UNVERIFIED`; do not claim the $0.502/hour bound covers an unapproved price. Unknown cost or failed shutdown verification ends the attempt.

The possible later grant is validated by the **unchanged** inner `validate_user_approval` and `verify_provider_observation` functions. Then the outer layer additionally requires:

- Exact returned pod, CA-MTL-1, A40, one GPU, Secure Cloud, exact storage and signed-in rates.
- The original outer deadline and stop time, with no reset after stop or resume.
- `inner.maximum_usd + accrued_upper_usd <= outer.maximum_usd <= 1.50`.
- Inner approval after genuine initial stopped verification; stopped observation after that approval; observation no older than 600 seconds at the resume request.
- The inner resume-request timestamp no more than five seconds old at the actual dispatch boundary, rechecked after independent reads, control preflight and guard verification.
- A new independent read of the same stopped pod at zero hourly charge, matching its first-bound physical machine and all allocation fields.
- Successful existing exact-pod stopped control preflight, plus a still-live external guard for the original intent/deadlines. This uses a synthetic verifier interface in tests; no live verifier is claimed here.

Write a durable exclusive resume-attempt receipt before the one resume request. A failed or ambiguous response is terminal; do not retry or find another host/pod/GPU/region. Read the resumed pod and require the same physical machine ID. A same-pod ID with a changed physical machine fails and triggers stop. This deliberately preserves the user's no-replacement-host requirement even if the provider ordinarily reschedules pods.

There is no inference handoff in this milestone. Consequently, every successful simulated resume immediately stops again after identity verification. A future reviewed integration must retain the same outer intent/budget/deadline and all unchanged inner cache, token, GPU, process-supervisor, per-generation watchdog, raw/parse ledger, and independent-decoding gates. It must not manufacture an earlier stopped receipt or reuse a stale observation.

## Unresolved live prerequisites

1. A provider adapter that can verify every required readback field and immutable bootstrap settings, has no retry behavior for create/resume, and independently supervises bounded requests.
2. A separately deployed external shutdown worker and trustworthy liveness verifier that survive allocator death, reconcile lost create responses, and stop only the uniquely owned pod at the fixed outer stop time. Receipt shape checking in local tests is not that verifier.
3. Inspection of the pinned image's inherited NVIDIA entrypoint/hooks. Sleep-only CMD alone does not prove bootstrap has no other effects.
4. A reviewed handoff from the outer ledger to the unchanged inner live runner that prevents bypass of cumulative accounting and preserves its control preflight/pod-watchdog/supervisor guarantees.
5. Exact operation-specific authorization after current provider facts are known, plus final independent provider stopped/$0-hour verification and temporary credential revocation when a real operation eventually completes.
6. A complete pinned Qwen cache and the unchanged input/token artifacts, made available through separately reviewed preparation compatible with the inner runner's no-automatic-download policy. This initial create-and-stop phase does not download, inspect, or load model weights; no complete live cache is established by these tests.

No key is read by this module. No credentials, actual user grant, model cache or provider state are created by implementation or tests.
