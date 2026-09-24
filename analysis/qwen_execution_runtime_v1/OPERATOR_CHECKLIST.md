# Future live operator checklist

This checklist is for a later separately authorized live stage. No step here has been performed by the local implementation milestone.

- [ ] Verify the exact execution-plan SHA, prepared-plan SHA, resource proposal SHA, and current proposal validity.
- [ ] Confirm Secure Cloud; exact pod ID; one NVIDIA RTX 6000 Ada Generation; 80 GB container disk; zero persistent volume; no network volume; actual rates within ceilings.
- [ ] Create the exact user approval record from `PROTOCOL.md` outside Git with mode `0600`.
- [ ] While the exact pod is stopped, perform read/stop/read-back preflight using the same CLI and credential context; save its private receipt.
- [ ] Only after the separate spending decision, start that exact pod and set `RUNPOD_POD_ID`.
- [ ] Start the pinned watchdog before cache/model access; verify its private receipt and live process.
- [ ] Run the exact prepared plan with `--execute` only after the runtime policy is separately enabled and reviewed; the launcher supervises and can kill/reap the inference child group at the fixed hard timeout.
- [ ] Stop immediately after success or failure; preserve and hash artifacts where possible.
- [ ] Independently verify exact pod state is stopped/exited and billing is $0/hour.
- [ ] Revoke any temporary provider control credential.
- [ ] Keep failed/partial attempts incomplete; do not retry, resume, repair, or expand scope.
