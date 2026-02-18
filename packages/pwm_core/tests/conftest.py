"""Shared pytest fixtures for pwm_core tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure THIS repo's pwm_core is importable (not a stale pip install from
# another workspace).  We insert packages/pwm_core/ at the front of sys.path
# so that ``import pwm_core.targeting`` resolves to the local code.
_PKG_DIR = str(Path(__file__).resolve().parents[1])  # packages/pwm_core/
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

# Also invalidate any cached pwm_core module so Python re-discovers it
# from the updated sys.path.
for key in list(sys.modules):
    if key == "pwm_core" or key.startswith("pwm_core."):
        del sys.modules[key]

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
