"""Contract fuzzing tests — random-but-valid inputs must never crash agents."""

import os
import sys
import pytest
import numpy as np

PKG_ROOT = os.path.join(os.path.dirname(__file__), "..", "pwm_core")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestContractFuzzing:
    """Generate random-but-bounded inputs and verify contracts validate."""

    def test_photon_report_rejects_nan(self):
        """PhotonReport must reject NaN in float fields."""
        from pwm_core.agents.contracts import PhotonReport, NoiseRegime
        with pytest.raises(Exception):
            PhotonReport(
                n_photons_per_pixel=float('nan'),
                snr_db=20.0,
                noise_regime=NoiseRegime.shot_limited,
                shot_noise_sigma=10.0,
                read_noise_sigma=3.0,
                total_noise_sigma=10.4,
                feasible=True,
                quality_tier="acceptable",
                throughput_chain=[],
                noise_model="poisson",
            )

    def test_photon_report_rejects_inf(self):
        """PhotonReport must reject Inf in float fields."""
        from pwm_core.agents.contracts import PhotonReport, NoiseRegime
        with pytest.raises(Exception):
            PhotonReport(
                n_photons_per_pixel=float('inf'),
                snr_db=20.0,
                noise_regime=NoiseRegime.shot_limited,
                shot_noise_sigma=10.0,
                read_noise_sigma=3.0,
                total_noise_sigma=10.4,
                feasible=True,
                quality_tier="acceptable",
                throughput_chain=[],
                noise_model="poisson",
            )

    def test_photon_report_rejects_extra_fields(self):
        """StrictBaseModel must reject extra fields."""
        from pwm_core.agents.contracts import PhotonReport, NoiseRegime
        with pytest.raises(Exception):
            PhotonReport(
                n_photons_per_pixel=1000.0,
                snr_db=20.0,
                noise_regime=NoiseRegime.shot_limited,
                shot_noise_sigma=10.0,
                read_noise_sigma=3.0,
                total_noise_sigma=10.4,
                feasible=True,
                quality_tier="acceptable",
                throughput_chain=[],
                noise_model="poisson",
                bogus_field="should_fail",
            )

    def test_bottleneck_scores_bounded(self):
        """Scores outside [0,1] must be rejected."""
        from pwm_core.agents.contracts import BottleneckScores
        with pytest.raises(Exception):
            BottleneckScores(photon=1.5, mismatch=0.5, compression=0.3, solver=0.2)

    def test_recoverability_report_valid(self):
        """A valid RecoverabilityReport must pass."""
        from pwm_core.agents.contracts import RecoverabilityReport, NoiseRegime, SignalPriorClass
        report = RecoverabilityReport(
            compression_ratio=0.036,
            noise_regime=NoiseRegime.shot_limited,
            signal_prior_class=SignalPriorClass.joint_spatio_spectral,
            operator_diversity_score=0.7,
            condition_number_proxy=1.5,
            recoverability_score=0.82,
            recoverability_confidence=0.95,
            expected_psnr_db=34.8,
            expected_psnr_uncertainty_db=0.5,
            recommended_solver_family="mst",
            verdict="excellent",
        )
        assert report.recoverability_score == 0.82

    def test_random_photon_reports(self):
        """Generate 100 random valid PhotonReports."""
        from pwm_core.agents.contracts import PhotonReport, NoiseRegime
        rng = np.random.default_rng(42)
        regimes = list(NoiseRegime)
        tiers = ["excellent", "acceptable", "marginal", "insufficient"]
        models = ["poisson", "gaussian", "mixed_poisson_gaussian"]

        for _ in range(100):
            n = rng.uniform(1, 1e6)
            snr = rng.uniform(-10, 60)
            report = PhotonReport(
                n_photons_per_pixel=float(n),
                snr_db=float(snr),
                noise_regime=regimes[int(rng.integers(len(regimes)))],
                shot_noise_sigma=float(rng.uniform(0, 1000)),
                read_noise_sigma=float(rng.uniform(0, 100)),
                total_noise_sigma=float(rng.uniform(0.01, 1000)),
                feasible=bool(rng.integers(2)),
                quality_tier=tiers[int(rng.integers(len(tiers)))],
                throughput_chain=[],
                noise_model=models[int(rng.integers(len(models)))],
            )
            # Round-trip serialization
            data = report.model_dump()
            PhotonReport.model_validate(data)

    def test_imaging_system_requires_detector(self):
        """ImagingSystem must require at least one detector."""
        from pwm_core.agents.contracts import ImagingSystem, ElementSpec, TransferKind, NoiseKind, ForwardModelType
        with pytest.raises(Exception):
            ImagingSystem(
                modality_key="test",
                elements=[
                    ElementSpec(
                        name="Source",
                        element_type="source",
                        transfer_kind=TransferKind.identity,
                    ),
                ],
                forward_model_type=ForwardModelType.linear_operator,
                forward_model_equation="y = Ax",
                signal_dims={"x": [64, 64], "y": [64, 64]},
            )

    def test_plan_intent_roundtrip(self):
        """PlanIntent round-trip serialization."""
        from pwm_core.agents.contracts import PlanIntent, ModeRequested, OperatorType
        intent = PlanIntent(
            mode_requested=ModeRequested.auto,
            has_measured_y=False,
            has_operator_A=False,
            operator_type=OperatorType.unknown,
            user_prompt="simulate CASSI imaging",
        )
        data = intent.model_dump()
        restored = PlanIntent.model_validate(data)
        assert restored.user_prompt == intent.user_prompt
