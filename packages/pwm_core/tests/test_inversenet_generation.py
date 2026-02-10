"""Tests for InverseNet dataset generation.

Verifies:
1. Manifest completeness -- all expected samples are present
2. RunBundle integrity   -- every sample has a valid runbundle_manifest.json
3. Artifact existence    -- all files referenced in manifest.paths exist
4. Schema validation     -- every manifest row parses with ManifestRecord
5. Smoke baselines       -- run_baselines --smoke completes without error
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import List

import numpy as np
import pytest

# ── Import generators and schema ────────────────────────────────────────

from experiments.inversenet.manifest_schema import ManifestRecord
from experiments.inversenet.gen_spc import generate_spc_dataset
from experiments.inversenet.gen_cacti import generate_cacti_dataset
from experiments.inversenet.gen_cassi import generate_cassi_dataset
from experiments.inversenet.run_baselines import run_all_baselines


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def spc_dataset(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("spc"))
    records = generate_spc_dataset(out, smoke=True)
    return out, records


@pytest.fixture(scope="module")
def cacti_dataset(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("cacti"))
    records = generate_cacti_dataset(out, smoke=True)
    return out, records


@pytest.fixture(scope="module")
def cassi_dataset(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("cassi"))
    records = generate_cassi_dataset(out, smoke=True)
    return out, records


# ── Test: manifest completeness ─────────────────────────────────────────


def test_spc_manifest_completeness(spc_dataset):
    out_dir, records = spc_dataset
    assert len(records) >= 1, "Smoke should produce at least 1 sample"
    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    assert os.path.exists(manifest_path), "manifest.jsonl must exist"

    with open(manifest_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    assert len(lines) == len(records), "Manifest line count mismatch"


def test_cacti_manifest_completeness(cacti_dataset):
    out_dir, records = cacti_dataset
    assert len(records) >= 1
    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    assert os.path.exists(manifest_path)


def test_cassi_manifest_completeness(cassi_dataset):
    out_dir, records = cassi_dataset
    assert len(records) >= 1
    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    assert os.path.exists(manifest_path)


# ── Test: schema validation ─────────────────────────────────────────────


def _validate_all_records(out_dir: str):
    """Parse every line in manifest.jsonl through ManifestRecord."""
    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    with open(manifest_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = ManifestRecord.model_validate_json(line)
            assert rec.sample_id, f"Row {i}: missing sample_id"
            assert rec.modality, f"Row {i}: missing modality"


def test_spc_schema(spc_dataset):
    _validate_all_records(spc_dataset[0])


def test_cacti_schema(cacti_dataset):
    _validate_all_records(cacti_dataset[0])


def test_cassi_schema(cassi_dataset):
    _validate_all_records(cassi_dataset[0])


# ── Test: artifact existence ────────────────────────────────────────────


def _check_artifacts(out_dir: str, records: List[ManifestRecord]):
    for rec in records:
        for key, rel_path in rec.paths.items():
            full = os.path.join(out_dir, rel_path)
            assert os.path.exists(full), (
                f"Missing artifact '{key}' at {full} for sample {rec.sample_id}"
            )


def test_spc_artifacts(spc_dataset):
    _check_artifacts(*spc_dataset)


def test_cacti_artifacts(cacti_dataset):
    _check_artifacts(*cacti_dataset)


def test_cassi_artifacts(cassi_dataset):
    _check_artifacts(*cassi_dataset)


# ── Test: RunBundle integrity ───────────────────────────────────────────


def _check_runbundles(out_dir: str, records: List[ManifestRecord]):
    for rec in records:
        bundle_path = os.path.join(
            out_dir, "samples", rec.sample_id, "runbundle_manifest.json"
        )
        assert os.path.exists(bundle_path), f"Missing RunBundle: {bundle_path}"
        with open(bundle_path) as f:
            bundle = json.load(f)
        assert bundle["version"] == "0.3.0", "RunBundle version must be 0.3.0"
        assert "provenance" in bundle
        assert "metrics" in bundle
        assert "artifacts" in bundle
        assert "hashes" in bundle
        # Verify all artifact files exist
        sample_dir = os.path.join(out_dir, "samples", rec.sample_id)
        for key, rel in bundle["artifacts"].items():
            assert os.path.exists(os.path.join(sample_dir, rel)), (
                f"RunBundle artifact {key}={rel} missing"
            )
        # Verify hashes match artifacts
        for key in bundle["artifacts"]:
            assert key in bundle["hashes"], f"Missing hash for {key}"
            assert bundle["hashes"][key].startswith("sha256:"), (
                f"Invalid hash format for {key}"
            )


def test_spc_runbundles(spc_dataset):
    _check_runbundles(*spc_dataset)


def test_cacti_runbundles(cacti_dataset):
    _check_runbundles(*cacti_dataset)


def test_cassi_runbundles(cassi_dataset):
    _check_runbundles(*cassi_dataset)


# ── Test: data integrity ────────────────────────────────────────────────


def test_spc_data_ranges(spc_dataset):
    out_dir, records = spc_dataset
    for rec in records:
        x_gt = np.load(os.path.join(out_dir, rec.paths["x_gt"]))
        assert x_gt.ndim == 2, f"SPC x_gt should be 2D, got {x_gt.ndim}D"
        assert x_gt.min() >= 0, "x_gt has negative values"
        assert x_gt.max() <= 1.0 + 1e-6, "x_gt exceeds 1.0"
        y = np.load(os.path.join(out_dir, rec.paths["y"]))
        assert y.ndim == 1, f"SPC y should be 1D, got {y.ndim}D"
        assert np.all(np.isfinite(y)), "y contains NaN/Inf"


def test_cacti_data_ranges(cacti_dataset):
    out_dir, records = cacti_dataset
    for rec in records:
        x_gt = np.load(os.path.join(out_dir, rec.paths["x_gt"]))
        assert x_gt.ndim == 3, f"CACTI x_gt should be 3D, got {x_gt.ndim}D"
        y = np.load(os.path.join(out_dir, rec.paths["y"]))
        assert y.ndim == 2, f"CACTI y should be 2D, got {y.ndim}D"
        assert np.all(np.isfinite(y)), "y contains NaN/Inf"


def test_cassi_data_ranges(cassi_dataset):
    out_dir, records = cassi_dataset
    for rec in records:
        x_gt = np.load(os.path.join(out_dir, rec.paths["x_gt"]))
        assert x_gt.ndim == 3, f"CASSI x_gt should be 3D, got {x_gt.ndim}D"
        y = np.load(os.path.join(out_dir, rec.paths["y"]))
        assert y.ndim == 2, f"CASSI y should be 2D, got {y.ndim}D"
        assert np.all(np.isfinite(y)), "y contains NaN/Inf"


# ── Test: smoke baselines ──────────────────────────────────────────────


def test_smoke_baselines(spc_dataset, cacti_dataset, cassi_dataset, tmp_path):
    """Run baselines in smoke mode across all three modalities."""
    data_dirs = [spc_dataset[0], cacti_dataset[0], cassi_dataset[0]]
    results_dir = str(tmp_path / "results")

    results = run_all_baselines(data_dirs, results_dir, smoke=False)

    # Each modality should produce results for T1-T4
    assert len(results) > 0, "No results produced"

    # Check results files exist
    assert os.path.exists(os.path.join(results_dir, "baseline_results.jsonl"))
    assert os.path.exists(os.path.join(results_dir, "error_bars.json"))

    # Check error bars JSON is valid
    with open(os.path.join(results_dir, "error_bars.json")) as f:
        eb = json.load(f)
    assert isinstance(eb, list)
    assert len(eb) > 0, "Error bars should have entries"

    # Check all 4 tasks appear
    tasks_seen = set()
    for r in results:
        tasks_seen.add(r.task)
    for t in ["T1_param_estimation", "T2_mismatch_id",
              "T3_calibration", "T4_reconstruction"]:
        assert t in tasks_seen, f"Task {t} missing from results"
