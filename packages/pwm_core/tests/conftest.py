"""Shared pytest fixtures for pwm_core tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CONTRIB = Path(__file__).resolve().parent.parent / "contrib"
DATASETS = CONTRIB / "datasets"


@pytest.fixture
def tiny_spc_dataset():
    """Load the tiny SPC demo dataset for smoke tests."""
    path = DATASETS / "examples" / "tiny_spc_demo_v1.npz"
    if path.exists():
        return np.load(str(path))
    # Generate synthetic fallback
    rng = np.random.default_rng(42)
    return {
        "y": rng.random((32, 32)).astype(np.float32),
        "mask": (rng.random((32, 32)) > 0.5).astype(np.float32),
    }


@pytest.fixture
def tiny_cassi_cube():
    """64x64x8 synthetic CASSI datacube."""
    return np.random.default_rng(42).random((64, 64, 8)).astype(np.float32)


@pytest.fixture
def tiny_widefield_image():
    """64x64 synthetic widefield image."""
    return np.random.default_rng(42).random((64, 64)).astype(np.float32)


@pytest.fixture
def sample_manifest(tmp_path):
    """Create a valid sample manifest for testing."""
    import json

    # Create a dummy sample file
    sample_data = np.random.default_rng(42).random((16, 16)).astype(np.float32)
    sample_path = tmp_path / "sample_001.npy"
    np.save(str(sample_path), sample_data)

    from pwm_core.io.retrieval import compute_sha256
    sha = compute_sha256(str(sample_path))

    manifest = {
        "dataset_id": "test_dataset_v1",
        "modality": "widefield",
        "license": "MIT",
        "samples": [
            {
                "id": "sample_001",
                "path": str(sample_path),
                "split": "train",
                "sha256": sha,
            }
        ],
    }

    manifest_path = tmp_path / "manifest.json"
    with open(str(manifest_path), "w") as f:
        json.dump(manifest, f)

    return str(manifest_path)
