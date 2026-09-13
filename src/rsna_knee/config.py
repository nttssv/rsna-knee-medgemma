from dataclasses import dataclass, asdict, fields
from pathlib import Path
import os
import tomllib


@dataclass(frozen=True)
class Experiment:
    model: str
    revision: str
    seed: int = 42
    max_studies: int = 80
    holdout_fraction: float = 0.25
    slices_per_plane: int = 2
    image_size: int = 448
    max_steps: int = 20
    accumulation: int = 4
    lora_rank: int = 8
    learning_rate: float = 5e-5
    checkpoint_every: int = 5
    minimum_gpu_gib: float = 37
    export_train_targets: bool = True

    def __post_init__(self):
        for name in (
            "max_steps",
            "accumulation",
            "lora_rank",
            "checkpoint_every",
            "image_size",
            "slices_per_plane",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.holdout_fraction < 1:
            raise ValueError("Invalid holdout fraction")
        if self.max_studies < 12:
            raise ValueError("max_studies must be at least 12")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not self.revision:
            raise ValueError("Pin a base model revision for reproducibility")

    def identity(self):
        # Extending a completed run or changing save frequency does not reset training.
        return {
            k: v
            for k, v in asdict(self).items()
            if k
            not in {
                "max_steps",
                "checkpoint_every",
                "minimum_gpu_gib",
                "export_train_targets",
            }
        }


def load(path, state_dir=None):
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    known = {f.name for f in fields(Experiment)}
    unknown = set(raw.get("experiment", {})) - known
    if unknown:
        raise ValueError(f"Unknown experiment options: {sorted(unknown)}")
    cfg = Experiment(**raw["experiment"])
    root = (
        Path(state_dir or os.environ.get("RSNA_STATE_DIR", "state"))
        .expanduser()
        .resolve()
    )
    paths = {
        key: (root / Path(raw.get("paths", {}).get(key, key))).resolve()
        for key in ["data", "cache", "runs"]
    }
    return cfg, paths, root
