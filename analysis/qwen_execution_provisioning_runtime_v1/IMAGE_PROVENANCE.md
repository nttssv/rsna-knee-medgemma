# Bootstrap image provenance

The Docker Hub registry was queried read-only on 2026-09-24 UTC. Only manifest and configuration JSON were retrieved; no image layers or model weights were downloaded.

| Field | Observed value |
|---|---|
| Descriptive tag | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| OCI index digest | `sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35` |
| Linux amd64 manifest | `sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851` |
| Image config digest | `sha256:02b731f844d2fbc4dd0de87253081363864327c7931b1ae2b5daa60abc600c51` |
| Configured entrypoint | `["/opt/nvidia/nvidia_entrypoint.sh"]` |
| Default command | `["/start.sh"]` |
| Compressed image-layer sizes, summed | 10,564,340,940 bytes |

The proposed image reference uses the immutable Linux amd64 manifest, not the mutable tag or a mutable provider template. Its initial command must be explicitly bound to the reviewed sleep-only command. A provider adapter must verify that its command setting actually replaces the default `/start.sh` command and that the returned allocation uses the expected image and command.

Replacing the command does **not** replace the NVIDIA entrypoint. Manifest inspection alone does not inspect that script or its hooks. Verifying its behavior, the provider's command mapping, and absence of unintended startup/model activity is still required before any live enablement. The local controller's synthetic tests are not evidence of those live properties.

The layer-size total is provenance only. It is neither an allocation of local disk nor a model-cache readiness result. The complete pinned Qwen cache remains a separate prerequisite for later inference.

Sources: the [Docker Hub image repository](https://hub.docker.com/r/runpod/pytorch) and its public registry manifest/config endpoints under `https://registry-1.docker.io/v2/runpod/pytorch/`. Anonymous registry authorization was used only in memory and was not saved in this repository.
