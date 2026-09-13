"""Atomic checkpoints for adapters, optimizer state and random generators."""

from pathlib import Path
import hashlib
import json
import os
import random
import shutil


def fingerprint(configuration, split_bytes, input_bytes, labels_bytes, code_bytes):
    h = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode())
    for value in (split_bytes, input_bytes, labels_bytes, code_bytes):
        h.update(len(value).to_bytes(8, "big"))
        h.update(value)
    return h.hexdigest()


def save(root, step, model, optimizer, sampler, history, identity):
    import torch
    import numpy as np

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"step-{step:06d}"
    if target.exists():
        raise FileExistsError(f"Checkpoint already exists: {target}")
    staging = root / f".step-{step:06d}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    model.save_pretrained(staging / "adapter", safe_serialization=True)
    ns = np.random.get_state()
    state = dict(
        step=step,
        optimizer=optimizer.state_dict(),
        sampler=sampler.getstate(),
        python_rng=random.getstate(),
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        numpy_rng=[ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]],
        history=history,
        identity=identity,
    )
    torch.save(state, staging / "trainer_state.pt")
    (staging / "complete.json").write_text(
        json.dumps({"step": step, "identity": identity})
    )
    staging.rename(target)
    temporary = root / ".latest.partial"
    temporary.write_text(json.dumps({"checkpoint": target.name, "step": step}))
    os.replace(temporary, root / "latest.json")
    return target


def resolve(path):
    path = Path(path)
    if (path / "latest.json").is_file():
        name = json.loads((path / "latest.json").read_text())["checkpoint"]
        if Path(name).name != name:
            raise ValueError("Unsafe checkpoint pointer")
        path = path / name
    if (
        not (path / "complete.json").is_file()
        or not (path / "trainer_state.pt").is_file()
    ):
        raise ValueError(
            "Not a complete training checkpoint. Legacy adapters require --init-adapter."
        )
    return path


def restore(path, optimizer, sampler, identity):
    import torch
    import numpy as np

    state = torch.load(
        resolve(path) / "trainer_state.pt", map_location="cpu", weights_only=True
    )
    if state["identity"] != identity:
        raise ValueError(
            "Checkpoint/configuration, split, labels, preprocessing, or images differ"
        )
    optimizer.load_state_dict(state["optimizer"])
    sampler.setstate(state["sampler"])
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA state requires a CUDA device")
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    ns = state["numpy_rng"]
    np.random.set_state(
        (ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4])
    )
    return state["step"], state["history"]
