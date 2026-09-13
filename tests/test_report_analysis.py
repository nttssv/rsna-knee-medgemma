"""Portability/integrity checks with synthetic split metadata only."""

from pathlib import Path
import importlib.util
import json
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("report_reproduction", ROOT / "analysis/report_labeling/reproduce.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture_snapshot(tmp_path):
    code = tmp_path / "code"
    snapshot = tmp_path / "private"
    (code / "src").mkdir(parents=True)
    snapshot.mkdir()
    (code / "src/pipeline.py").write_text("# synthetic code\n")
    (snapshot / "splits.csv").write_text("StudyInstanceUID,split\nsynthetic,development\n")
    (snapshot / "split_manifest.json").write_text('{"synthetic":true}')
    hashes = {"src/pipeline.py": MODULE.sha(code / "src/pipeline.py")}
    (code / "frozen_code_manifest.json").write_text(json.dumps({"files_sha256": hashes}))
    private_hashes = {name: MODULE.sha(snapshot / name) for name in MODULE.PRIVATE_INPUTS}
    (snapshot / "freeze_manifest.json").write_text(json.dumps({"files_sha256": hashes | private_hashes}))
    return code, snapshot


def test_recorded_frozen_sources_unchanged():
    assert len(MODULE.verify_code()["files_sha256"]) == 10


def test_experiment_assembly_relocates_and_preserves_hashes(tmp_path):
    code, snapshot = fixture_snapshot(tmp_path)
    target = tmp_path / "other-provider" / "experiment"
    frozen = MODULE.prepare_experiment(code, snapshot, target)
    assert all(MODULE.sha(target / name) == digest for name, digest in frozen["files_sha256"].items())
    with pytest.raises(FileExistsError):
        MODULE.prepare_experiment(code, snapshot, target)


def test_changed_code_and_private_split_are_rejected(tmp_path):
    code, snapshot = fixture_snapshot(tmp_path)
    (code / "src/pipeline.py").write_text("# modified\n")
    with pytest.raises(ValueError, match="Frozen code differs"):
        MODULE.prepare_experiment(code, snapshot, tmp_path / "output")
    (code / "src/pipeline.py").write_text("# synthetic code\n")
    (snapshot / "splits.csv").write_text("changed")
    with pytest.raises(ValueError, match="Snapshot input differs"):
        MODULE.prepare_experiment(code, snapshot, tmp_path / "output")


def test_manifest_cannot_escape_root(tmp_path):
    with pytest.raises(ValueError, match="Unsafe path"):
        MODULE.relative_name("../outside.csv")
    code, snapshot = fixture_snapshot(tmp_path)
    p = snapshot / "freeze_manifest.json"
    m = json.loads(p.read_text())
    m["files_sha256"]["../outside.csv"] = "invalid"
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="unexpected set"):
        MODULE.prepare_experiment(code, snapshot, tmp_path / "output")


def test_reproduced_results_must_match_reference(tmp_path):
    snapshot = tmp_path / "snapshot"
    output = tmp_path / "output"
    snapshot.mkdir(); output.mkdir()
    p = output / "metrics.csv"
    p.write_text("count\n2\n")
    (snapshot / "evaluation_manifest.json").write_text(json.dumps({"output_sha256": {"metrics.csv": MODULE.sha(p)}}))
    assert MODULE.compare_results(snapshot, output) == {"metrics.csv": True}
    p.write_text("count\n3\n")
    with pytest.raises(ValueError, match="differ"):
        MODULE.compare_results(snapshot, output)
    p.unlink()
    with pytest.raises(ValueError, match="filenames differ"):
        MODULE.compare_results(snapshot, output)
    p.write_text("count\n2\n")
    (output / "extra.csv").write_text("count\n1\n")
    with pytest.raises(ValueError, match="filenames differ"):
        MODULE.compare_results(snapshot, output)
