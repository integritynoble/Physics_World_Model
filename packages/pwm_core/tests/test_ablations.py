"""test_ablations.py

Verify that each ablation produces measurable degradation (>0.5 dB or
metric drop) across all 3 depth modalities.

Tests: 4 ablations x 3 modalities = 12 test cases.
"""

from __future__ import annotations

import pytest
import numpy as np

from experiments.pwm_flagship.ablations import (
    _run_full_pipeline,
    _ablation_no_photon,
    _ablation_no_recoverability,
    _ablation_no_mismatch,
    _ablation_no_runbundle,
    MODALITIES,
    ABLATION_FNS,
    ABLATION_NAMES,
)

BASE_SEED = 6000
RECON_ITERS = 15  # fast iterations for tests
MIN_DEGRADATION_DB = 0.5


# ---------------------------------------------------------------------------
# Parametrize: 4 ablations x 3 modalities
# ---------------------------------------------------------------------------

ABLATION_MODALITY_PAIRS = [
    (ablation, modality)
    for ablation in ABLATION_NAMES
    for modality in MODALITIES
]

IDS = [f"{abl}_{mod}" for abl, mod in ABLATION_MODALITY_PAIRS]


class TestAblations:
    """Each ablation must produce measurable degradation."""

    @pytest.mark.parametrize(
        "ablation_name,modality",
        ABLATION_MODALITY_PAIRS,
        ids=IDS,
    )
    def test_ablation_causes_degradation(
        self, ablation_name: str, modality: str
    ) -> None:
        """Removing a component must degrade PSNR by >= 0.5 dB."""
        n_trials = 3
        seed = BASE_SEED

        # Baseline: full pipeline with all components
        baseline_psnrs = []
        for trial in range(n_trials):
            s = seed + trial * 7
            r = _run_full_pipeline(modality, s, RECON_ITERS)
            baseline_psnrs.append(r["psnr"])
        baseline_mean = float(np.mean(baseline_psnrs))

        # Ablation: remove one component
        ablation_fn = ABLATION_FNS[ablation_name]
        ablation_psnrs = []
        for trial in range(n_trials):
            s = seed + trial * 7
            r = ablation_fn(modality, s, RECON_ITERS)
            ablation_psnrs.append(r["psnr"])
        ablation_mean = float(np.mean(ablation_psnrs))

        degradation = baseline_mean - ablation_mean

        assert degradation >= MIN_DEGRADATION_DB, (
            f"Ablation '{ablation_name}' on '{modality}' should cause "
            f">= {MIN_DEGRADATION_DB} dB degradation, got {degradation:.2f} dB. "
            f"Baseline={baseline_mean:.2f}, Ablation={ablation_mean:.2f}"
        )

    def test_all_modalities_covered(self) -> None:
        """Verify we test all 3 modalities."""
        assert set(MODALITIES) == {"spc", "cacti", "cassi"}

    def test_all_ablations_covered(self) -> None:
        """Verify we test all 4 ablations."""
        assert len(ABLATION_NAMES) == 4
        assert set(ABLATION_NAMES) == {
            "no_photon", "no_recoverability",
            "no_mismatch", "no_runbundle",
        }
