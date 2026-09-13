from dataclasses import replace
from pathlib import Path
import ast
import json
import numpy as np
import pandas as pd
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage
from rsna_knee import imaging
from rsna_knee.audit import run
from rsna_knee.config import load
from rsna_knee.constants import LABELS, PLANES

ROOT = Path(__file__).resolve().parents[1]


def test_preprocessing_matches_reference_notebook():
    nb = json.loads((ROOT / "notebooks/medgemma_pilot_reference.ipynb").read_text())
    reference = {}
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = cell["source"]
            source = "".join(source) if isinstance(source, list) else source
            reference.update(
                {
                    n.name: ast.dump(n, include_attributes=False)
                    for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef)
                }
            )
    for node in ast.parse(Path(imaging.__file__).read_text()).body:
        if isinstance(node, ast.FunctionDef):
            assert ast.dump(node, include_attributes=False) == reference[node.name]


def test_audit_and_saved_split_survive_directory_move(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    rows = []
    descriptors = []
    for i in range(12):
        uid = f"1.2.826.0.1.{i + 1}"
        row = {"StudyInstanceUID": uid, "Report": f"Synthetic report {i}"}
        row.update({label: ((i + j) % 2) for j, label in enumerate(LABELS)})
        if i == 0:
            row["ACL"] = np.nan
        rows.append(row)
        for j, plane in enumerate(PLANES):
            series = f"{uid}.{j + 1}"
            folder = data / "train_series" / uid / series
            folder.mkdir(parents=True)
            descriptors.append(
                dict(
                    StudyInstanceUID=uid,
                    SeriesInstanceUID=series,
                    Anatomical_Plane=plane,
                    Fluid_Sensitive=1,
                    Fat_Suppression=1,
                )
            )
            for k in range(2):
                meta = FileMetaDataset()
                meta.TransferSyntaxUID = ExplicitVRLittleEndian
                meta.MediaStorageSOPClassUID = MRImageStorage
                meta.MediaStorageSOPInstanceUID = f"{series}.{k + 1}"
                ds = FileDataset(
                    str(folder / f"{k}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128
                )
                ds.Rows = 8
                ds.Columns = 10
                ds.SamplesPerPixel = 1
                ds.PhotometricInterpretation = "MONOCHROME2"
                ds.BitsAllocated = 16
                ds.BitsStored = 16
                ds.HighBit = 15
                ds.PixelRepresentation = 0
                ds.ImagePositionPatient = [0, 0, k]
                ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
                ds.PixelSpacing = [1, 1]
                ds.PatientID = f"synthetic-{i}"
                ds.Manufacturer = "Synthetic"
                ds.PixelData = (
                    np.arange(80, dtype=np.uint16).reshape(8, 10) + k
                ).tobytes()
                ds.save_as(folder / f"{k}.dcm", enforce_file_format=True)
    pd.DataFrame(rows).to_csv(data / "train.csv", index=False)
    pd.DataFrame(descriptors).to_csv(data / "train_series.csv", index=False)
    cfg, _, _ = load(ROOT / "configs/pilot.toml", tmp_path)
    a = tmp_path / "run-a"
    b = tmp_path / "run-b"
    pilot, _ = run(cfg, data, a)
    moved = tmp_path / "other-provider"
    data.rename(moved)
    restored, _ = run(cfg, moved, b, a / "development_split.csv")
    pd.testing.assert_frame_equal(
        pilot[["StudyInstanceUID", "split", "group", "scanner_group"]],
        restored[["StudyInstanceUID", "split", "group", "scanner_group"]],
    )
    assert (a / "input_fingerprints.csv").read_bytes() == (
        b / "input_fingerprints.csv"
    ).read_bytes()
    assert pd.read_csv(b / "ground_truth.csv")["ACL"].isna().sum() == 1
    assert not (b / "ground_truth.csv").read_text().find("Synthetic report") >= 0
    from rsna_knee.review import export

    predictions = restored.loc[
        restored["split"] == "validation", ["StudyInstanceUID", "split"]
    ].copy()
    for label in LABELS:
        predictions[label] = 0.6
    predictions.to_csv(b / "validation_after.csv", index=False)
    export(cfg, moved, b)
    cases = json.loads((b / "case_review/cases.json").read_text())
    assert len(cases["cases"]) == len(predictions)
    assert all(c["before"]["ACL"] is None for c in cases["cases"])
    assert len(list((b / "case_review").rglob("slice-*.png"))) == 6 * len(predictions)
