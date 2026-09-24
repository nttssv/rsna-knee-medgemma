# Future operator checklist

This checklist is documentary only. Execution is disabled and this milestone makes no RunPod changes.

1. Verify the execution-plan SHA, prepared-plan SHA, prior plan SHA, alternate proposal SHA, source hashes, and CPU tests.
2. Select exactly one GPU from the versioned allowlist. In the signed-in provider UI, verify the exact existing pod ID, Secure Cloud, region, one matching GPU, 80 GB container disk, no persistent or network volume, and actual compute/storage rates. Confirm the same pod can be used before creating any approval. If unavailable, stop; no fallback is permitted.
3. Capture the exact provider-observation fields in a private `0600` file. Confirm its values match the proposed grant exactly.
4. Only after a separate spending decision, create the exact `0600` approval JSON using the actual rates and exact execution-plan SHA. Set the provider deadline no more than 60 minutes after start, and watchdog stop exactly 300 seconds earlier. Ensure the proposal quote is current.
5. While the exact pod is stopped, perform the reviewed same-pod stopped-state read/stop/read-back preflight and save the private receipt. Arm and independently verify the exact-pod watchdog before model/cache access.
6. Check exact GPU identity, one visible GPU, free VRAM, BF16, CUDA/Torch/package versions, driver provenance, offline model cache, SDPA, model revision, prepared token parity, context/input caps, and EOS IDs.
7. Run the fixed 20-generation maximum in ABBA order. The operator stops the whole experiment on any failure; no retry, fallback GPU/region/pod, or continuation is allowed. Stop the provider pod immediately after success/failure, independently verify stopped state and $0/hour, and revoke temporary control credentials.
