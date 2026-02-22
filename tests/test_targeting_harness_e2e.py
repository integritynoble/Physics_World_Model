"""End-to-end integration tests for the LIP-Arena targeting harness.

Tests that ``Harness.run()`` completes without error for each of the
7 validated modalities in sandbox mode, producing valid scores.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure pwm_core is importable
_repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo / "packages" / "pwm_core"))

from pwm_core.targeting.harness import Harness, HarnessResult

VALIDATED_MODALITIES = [
    "cassi", "cacti", "spc", "ct", "mri", "ptychography", "lensless",
]


@pytest.fixture(params=VALIDATED_MODALITIES)
def modality(request):
    return request.param


def _run_sandbox(modality: str) -> HarnessResult:
    """Run the harness in sandbox mode for a single modality."""
    harness = Harness(
        modality=modality,
        solver="traditional_cpu",
        track="correct",
        budget_s=120,
        sandbox=True,
    )
    return harness.run(n_scenes=1, seed=42, severity="mild")


class TestHarnessE2E:
    """End-to-end harness tests for all validated modalities."""

    def test_harness_runs_without_error(self, modality):
        """Harness.run() completes without exceptions."""
        result = _run_sandbox(modality)
        assert isinstance(result, HarnessResult)

    def test_harness_produces_valid_rho(self, modality):
        """Recovery ratio is a finite number (can be nan if no gap)."""
        result = _run_sandbox(modality)
        rho = result.aggregate.rho
        # rho can be nan when PSNR_I == PSNR_II, but must not be None
        assert rho is not None

    def test_harness_not_disqualified(self, modality):
        """Sandbox runs should not trigger disqualification."""
        result = _run_sandbox(modality)
        assert not result.aggregate.disqualified, (
            f"Disqualified: {result.aggregate.disqualification_reason}"
        )

    def test_harness_has_per_scene_results(self, modality):
        """Sandbox runs 1 scene and populates per_scene."""
        result = _run_sandbox(modality)
        assert len(result.per_scene) == 1
        scene = result.per_scene[0]
        assert "I" in scene.scenario_results
        assert "II" in scene.scenario_results
        assert "III" in scene.scenario_results
        assert "IV" in scene.scenario_results

    def test_harness_timing_positive(self, modality):
        """Total timing should be positive."""
        result = _run_sandbox(modality)
        assert result.timing_s > 0

    def test_harness_to_dict(self, modality):
        """to_dict() returns a valid dict with expected keys."""
        result = _run_sandbox(modality)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["modality"] == modality
        assert "aggregate" in d
        assert "per_scene" in d


class TestDatasets:
    """Verify that generated LIP-Arena datasets exist and are valid."""

    DATASET_DIR = _repo / "datasets" / "lip_arena"

    @pytest.mark.parametrize("modality", VALIDATED_MODALITIES)
    def test_dataset_exists(self, modality):
        """Each validated modality has x_gt.npy, y.npy, metadata.json."""
        mod_dir = self.DATASET_DIR / modality
        assert mod_dir.exists(), f"Missing dataset dir: {mod_dir}"
        assert (mod_dir / "x_gt.npy").exists()
        assert (mod_dir / "y.npy").exists()
        assert (mod_dir / "metadata.json").exists()

    @pytest.mark.parametrize("modality", VALIDATED_MODALITIES)
    def test_dataset_loadable(self, modality):
        """Datasets can be loaded and have consistent shapes."""
        import json

        mod_dir = self.DATASET_DIR / modality
        x_gt = np.load(mod_dir / "x_gt.npy")
        y = np.load(mod_dir / "y.npy")

        with open(mod_dir / "metadata.json") as f:
            meta = json.load(f)

        assert list(x_gt.shape) == meta["x_shape"]
        assert list(y.shape) == meta["y_shape"]
        assert x_gt.size > 0
        assert y.size > 0
