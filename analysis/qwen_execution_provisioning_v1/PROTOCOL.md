# Protocol: proposed create-and-stop preparation

This is a proposed operational extension, not an executable or authorized procedure. It must be implemented in a new version and reviewed before paid allocation. All frozen scientific and runtime files remain unchanged.

## Unchanged experiment

Consume the existing prepared package and the hashes in [provisioning_proposal.json](configs/provisioning_proposal.json). Qwen/Qwen3-14B remains pinned to `40c069824f4251a91eefaf281ebe4c544efd3e18`, BF16, SDPA, batch size 1. Use the same five development reports in control-1 → candidate-1 → candidate-2 → control-2 order, five reports per block, at most 20 generations. Preserve both prompts, both parsers, tokenization, 8,192 input / 2,048 output / 40,960 context limits, EOS IDs `[151645,151643]`, and the 40/18 split. No training, validation, full-development or bulk inference, retries, repair, or favorable-repeat selection.

## Before any creation

1. Refresh the signed-in allocation: Secure Cloud, CA-MTL-1, exactly one NVIDIA A40 with 48 GB, 80 GB container disk, zero persistent volume, null network volume, and actual rates no higher than $0.49/hour compute and $0.012/hour storage. Pin the deployment image and startup command; startup must not download weights, load a model, or run inference. Capacity is not a reservation.
2. Review and test a separate controller without provider mutations. It must arm outside the future pod before allocation, survive the allocating process failing, durably record the create attempt, bind the returned exact pod ID, and stop only that pod. There is no pod-local watchdog before a pod exists. Read-only access is insufficient for its stop duty; a temporary user-authorized control credential must be kept privately outside Git with minimum required provider permissions.
3. Resolve ambiguous create responses in advance: allow one create request only, no automatic HTTP retry. If the response is lost, use a reviewed read-only reconciliation method tied to the unique creation intent; do not issue another create or stop unrelated resources. An unresolved identity is a critical shutdown failure, not permission to guess a pod ID.
4. Obtain explicit authorization bound to this provisioning proposal, the one creation intent and exact resource boundary. A pod ID does not exist before creation: this authorization cannot pretend to be the frozen inference grant, which requires an exact existing ID. This milestone creates neither record.

## One outer clock and one cumulative budget

Let `first_create_requested_at` be a durable timezone-aware timestamp recorded immediately before the sole create request. Set `outer_provider_deadline` no later than that timestamp plus 3,600 seconds, and `outer_watchdog_stop_at = outer_provider_deadline - 300 seconds`. Arm the external controller against those fixed timestamps before dispatch. Every setup, stop, waiting interval, transfer, resume, inference, and final shutdown fits inside this same wall-clock window. Never restart its clock after stop or resume.

The initial creation and any later resume share a cumulative maximum of $1.50. Retain the actual compute/storage rates and charge evidence from both phases. Before any resume, reserve enough time and remaining budget for setup, the supervised inference worker, artifact copy, and shutdown. If that cannot be demonstrated, leave the pod stopped and end the attempt. Do not extend a deadline, increase a budget, attach a volume, or change host/GPU/region to recover it.

## Initial paid phase: create, identify, stop

After the above implementation, review, and authorization, a future operator may dispatch the one creation request. Billable provisioning starts here. Do not claim this phase was stopped-state preparation. Bind the exact returned pod identity and read back the allocation fields and rates immediately; any mismatch triggers stopping that exact pod and failure.

Request stop as soon as the exact pod can be controlled, without waiting for SSH, cache setup, model loading, or inference. A reviewed implementation must bound each stop request and retries inside the existing shutdown reserve; only shutdown retries are permitted. The independent external controller remains active as the deadline backstop. Copy control receipts when possible, then independently read the provider's stopped state and verify $0/hour. A stop command's success response alone is insufficient. Failed shutdown verification ends the experiment and requires operator intervention; it never permits inference.

## Stopped phase and possible later inference

Only genuine stopped-state evidence from the newly created exact pod may support the existing runner's preflight. Verify its resource fields, rates, actual pod ID, and region again. Perform exact-pod read/stop/read-back control preflight using the approved executable and credential path. No receipt from another pod is reusable. The existing stopped preflight does not prove a future live stop.

A later stage requires a separately reviewed implementation/enablement plan and explicit approval for the exact pod and plan hash. Capture a fresh stopped provider observation after that approval, no more than 600 seconds before its one resume request. If capacity has been released and resume fails, end the attempt: do not retry resume, create another pod, or select another GPU/host/region.

After a permitted resume, keep the independent external controller active until final shutdown has been externally verified. Arm and verify the approved pod-local watchdog before any cache/model access and recheck it before every generation. Both controllers use the same `outer_watchdog_stop_at`. Preserve all existing cache, hardware, dependency, token-parity, hard supervisor, durable receipt, and byte-exact independent token-decoding checks. The inference worker must finish before the watchdog time, under its existing strict sub-1,800-second hard limit and any shorter remaining time.

Stop immediately on completion or failure; preserve/hash/copy artifacts first only when time permits. Independently confirm stopped state and $0/hour. Revoke the temporary control credential after shutdown verification and retain only non-secret receipts. Do not disable the only working stop controller while stopped state remains unverified.

## Required implementation and review still missing

The frozen grant uses `provider_start_requested_at` for the budget/window calculation and also requires a stopped observation before that timestamp. It has no field for an earlier billable creation attempt or cumulative cost from that phase. Using the initial creation timestamp would put the later stopped observation after start and fail the reviewed observation ordering; using the later resume timestamp alone would omit initial provisioning from its estimate. A documentary outer deadline does not solve this binding gap.

A new, reviewed operational layer must bind the first-create timestamp, exact returned pod, immutable outer deadline/stop time, spent-to-date and remaining budget into the later authorization and controller provenance. It must prove that the existing inner runtime receives no later deadline and cannot bypass that cumulative accounting. This proposal does not claim that today's runner implements or accepts those new fields. Do not edit frozen grants to add them, fabricate a pre-creation stopped receipt, or reset the budget to make an existing gate pass.

Before closing implementation, CPU/synthetic tests must exercise one-create dispatch, no retry after ambiguous response, exact-pod identity reconciliation, pre-armed external timer, controller failure, create/stop/timeouts, resource mismatch, cumulative-budget exhaustion, unchanged outer deadline across stop/resume, fresh stopped observations, no GPU/region/host fallback, and retained watchdog/supervisor/token-decode protections. None of those new controller tests or paid actions is implemented by this documentary milestone.
