from dataclasses import replace
from pathlib import Path
import json
import random
import tarfile
import io
import numpy as np
import pandas as pd
import pytest
import torch
from rsna_knee.config import load
from rsna_knee import checkpoints, bundle
from rsna_knee.audit import validate_split

ROOT = Path(__file__).resolve().parents[1]


def test_config_moves_without_editing_code(tmp_path):
    cfg, paths, root = load(ROOT / "configs/pilot.toml", tmp_path / "provider-b")
    assert paths["data"] == tmp_path / "provider-b/data"
    assert cfg.identity() == replace(cfg, max_steps=40, checkpoint_every=10).identity()
    with pytest.raises(ValueError):
        replace(cfg, accumulation=0)


def test_split_rejects_leakage_and_duplicates():
    available = pd.DataFrame({"StudyInstanceUID": ["a", "b"]})
    split = pd.DataFrame(
        {
            "StudyInstanceUID": ["a", "b"],
            "split": ["train", "validation"],
            "group": ["same", "same"],
            "scanner_group": ["x", "x"],
        }
    )
    with pytest.raises(ValueError, match="leakage"):
        validate_split(split, available)
    split["group"] = ["x", "y"]
    validate_split(split, available)
    split["StudyInstanceUID"] = ["a", "a"]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_split(split, available)


class ToyAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 3), torch.nn.Dropout(0.25), torch.nn.Linear(3, 1)
        )

    def forward(self, x):
        return self.net(x)

    def save_pretrained(self, path, safe_serialization=True):
        Path(path).mkdir()
        torch.save(self.state_dict(), Path(path) / "toy.pt")


def update(model, optimizer, sampler):
    sample = sampler.randrange(10)
    x = torch.rand(4, 3) + sample
    loss = model(x).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.item(), sample


def test_checkpoint_resume_matches_uninterrupted_next_update(tmp_path):
    torch.manual_seed(71)
    random.seed(5)
    np.random.seed(6)
    model = ToyAdapter()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    sampler = random.Random(8)
    update(model, opt, sampler)
    saved = checkpoints.save(
        tmp_path, 1, model, opt, sampler, [{"step": 1, "loss": 0.2}], "fixed"
    )
    expected = update(model, opt, sampler)
    expected_weights = {k: v.clone() for k, v in model.state_dict().items()}
    restored = ToyAdapter()
    restored.load_state_dict(torch.load(saved / "adapter/toy.pt", weights_only=True))
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=0.01)
    restored_sampler = random.Random(999)
    step, history = checkpoints.restore(
        tmp_path, restored_opt, restored_sampler, "fixed"
    )
    assert step == 1 and len(history) == 1
    assert update(restored, restored_opt, restored_sampler) == expected
    for k, v in restored.state_dict().items():
        torch.testing.assert_close(v, expected_weights[k], rtol=0, atol=0)
    with pytest.raises(ValueError, match="differ"):
        checkpoints.restore(tmp_path, restored_opt, restored_sampler, "changed")


def test_partial_checkpoint_does_not_replace_latest(tmp_path):
    model = ToyAdapter()
    opt = torch.optim.AdamW(model.parameters())
    sampler = random.Random()
    saved = checkpoints.save(tmp_path, 1, model, opt, sampler, [], "a")

    def fail(*args, **kwargs):
        raise OSError("simulated interrupted write")

    model.save_pretrained = fail
    with pytest.raises(OSError):
        checkpoints.save(tmp_path, 2, model, opt, sampler, [], "a")
    assert checkpoints.resolve(tmp_path) == saved
    with pytest.raises(ValueError):
        checkpoints.resolve(tmp_path / ".step-000002.partial")


def test_bundle_restore_and_optional_data(tmp_path):
    state = tmp_path / "state"
    (state / "runs/pilot").mkdir(parents=True)
    (state / "data").mkdir()
    (state / "runs/pilot/metrics.json").write_text('{"loss":0.2}')
    (state / "runs/pilot/tokenizer_config.json").write_text('{"test":true}')
    (state / "data/train.csv").write_text("synthetic")
    (state / "data/image.dcm").write_bytes(b"synthetic-dicom")
    (state / "data/token.json").write_text("should not be copied")
    archive = tmp_path / "backup.tar.gz"
    bundle.create(state, archive)
    restored = tmp_path / "restored"
    bundle.restore(archive, restored)
    assert (restored / "runs/pilot/metrics.json").read_text() == '{"loss":0.2}'
    assert (restored / "runs/pilot/tokenizer_config.json").is_file()
    assert (
        not (restored / "data/image.dcm").exists()
        and not (restored / "data/token.json").exists()
    )
    with pytest.raises(FileExistsError):
        bundle.restore(archive, restored)


def test_bundle_rejects_path_traversal(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("../escape")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"bad"))
    with pytest.raises(ValueError, match="Unsafe"):
        bundle.restore(archive, tmp_path / "restore")
    assert not (tmp_path / "escape").exists()
