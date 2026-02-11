"""Tests for canonical noise application module and photon_db integration."""

import os
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pwm_core.noise.apply import apply_photon_noise, get_noise_recipe


CONTRIB_DIR = os.path.join(os.path.dirname(__file__), "..", "contrib")


# ── Backward compatibility ───────────────────────────────────────────────


class TestApplyPhotonNoiseBackwardCompat:
    """The shared function must reproduce the old copy-pasted behaviour."""

    def test_poisson_gaussian_default(self):
        """Default mode matches the old inline implementation."""
        rng_old = np.random.default_rng(42)
        rng_new = np.random.default_rng(42)

        y = np.random.default_rng(0).random((64, 64)).astype(np.float32) * 10.0
        photon_level = 1e4

        # Old inline implementation (copied from the original code)
        scale = photon_level / (np.abs(y).max() + 1e-10)
        y_scaled = np.maximum(y * scale, 0)
        y_old = rng_old.poisson(y_scaled).astype(np.float32)
        read_sigma = np.sqrt(photon_level) * 0.01
        y_old += rng_old.normal(0, read_sigma, size=y.shape).astype(np.float32)
        y_old /= scale

        # New canonical function
        y_new = apply_photon_noise(y, photon_level, rng_new)

        np.testing.assert_array_equal(
            y_old, y_new,
            err_msg="Shared noise function must be bit-identical to old code"
        )


# ── Noise model modes ───────────────────────────────────────────────────


class TestNoiseModels:
    """Each noise model should behave as expected."""

    def test_poisson_only_no_read_noise(self):
        """Poisson-only mode: no Gaussian read noise component."""
        rng = np.random.default_rng(7)
        y = np.ones((32, 32), dtype=np.float32) * 5.0
        noisy = apply_photon_noise(y, 1e5, rng, noise_model="poisson")
        # Should still be noisy (Poisson)
        assert not np.allclose(noisy, y, atol=1e-6), "Poisson noise should change signal"
        # Mean should be close to original (law of large numbers)
        assert abs(noisy.mean() - y.mean()) < 0.5

    def test_gaussian_only_mode(self):
        """Gaussian-only mode: sigma = signal_range / photon_level."""
        rng = np.random.default_rng(99)
        y = np.ones((100, 100), dtype=np.float32) * 2.0
        photon_level = 100.0  # sigma = 2.0/100 = 0.02
        noisy = apply_photon_noise(y, photon_level, rng, noise_model="gaussian")
        noise = noisy - y
        empirical_sigma = noise.std()
        expected_sigma = 2.0 / 100.0  # signal_range / photon_level
        assert abs(empirical_sigma - expected_sigma) < 0.005, (
            f"Gaussian sigma mismatch: got {empirical_sigma:.4f}, "
            f"expected ~{expected_sigma:.4f}"
        )

    def test_unknown_model_raises(self):
        """Unknown noise_model should raise ValueError."""
        rng = np.random.default_rng(0)
        y = np.ones(10, dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown noise_model"):
            apply_photon_noise(y, 1e4, rng, noise_model="banana")

    def test_poisson_gaussian_adds_read_noise(self):
        """Default model should add both Poisson and Gaussian components."""
        rng = np.random.default_rng(42)
        y = np.ones((50, 50), dtype=np.float32) * 3.0
        noisy = apply_photon_noise(y, 1e4, rng, noise_model="poisson_gaussian")
        assert not np.allclose(noisy, y)


# ── photon_db.yaml schema ───────────────────────────────────────────────


class TestPhotonDbSchema:
    """photon_db.yaml should be well-formed with noise_model + photon_levels."""

    @pytest.fixture
    def photon_db(self):
        path = os.path.join(CONTRIB_DIR, "photon_db.yaml")
        if not os.path.exists(path):
            pytest.skip("photon_db.yaml not found")
        with open(path) as f:
            return yaml.safe_load(f)

    def test_has_modalities(self, photon_db):
        assert "modalities" in photon_db
        assert len(photon_db["modalities"]) >= 18

    def test_all_modalities_have_noise_model(self, photon_db):
        for key, entry in photon_db["modalities"].items():
            assert "noise_model" in entry, f"{key} missing noise_model"
            assert entry["noise_model"] in (
                "poisson_gaussian", "poisson", "gaussian"
            ), f"{key} has invalid noise_model={entry['noise_model']}"

    def test_all_modalities_have_photon_levels(self, photon_db):
        for key, entry in photon_db["modalities"].items():
            assert "photon_levels" in entry, f"{key} missing photon_levels"
            levels = entry["photon_levels"]
            for level_name in ("bright", "standard", "low_light"):
                assert level_name in levels, (
                    f"{key} missing level '{level_name}'"
                )

    def test_photon_levels_have_required_fields(self, photon_db):
        for key, entry in photon_db["modalities"].items():
            for level_name, level in entry["photon_levels"].items():
                assert "scenario" in level, (
                    f"{key}/{level_name} missing 'scenario'"
                )
                # Must have either n_photons or sigma (for gaussian)
                has_photons = "n_photons" in level and level["n_photons"] is not None
                has_sigma = "sigma" in level and level["sigma"] is not None
                assert has_photons or has_sigma, (
                    f"{key}/{level_name} needs either n_photons or sigma"
                )


# ── get_noise_recipe ─────────────────────────────────────────────────────


class TestGetNoiseRecipe:
    """get_noise_recipe should look up correct values from photon_db."""

    @pytest.fixture
    def photon_db_dict(self):
        path = os.path.join(CONTRIB_DIR, "photon_db.yaml")
        if not os.path.exists(path):
            pytest.skip("photon_db.yaml not found")
        with open(path) as f:
            return yaml.safe_load(f)

    def test_cassi_standard_recipe(self, photon_db_dict):
        recipe = get_noise_recipe("cassi", "standard", photon_db_dict)
        assert recipe["noise_model"] == "poisson_gaussian"
        assert recipe["n_photons"] == 1e4
        assert "read_sigma_fraction" in recipe

    def test_missing_modality_raises(self, photon_db_dict):
        with pytest.raises(KeyError, match="not found in photon_db"):
            get_noise_recipe("nonexistent_modality", "standard", photon_db_dict)

    def test_missing_level_raises(self, photon_db_dict):
        with pytest.raises(KeyError, match="not found for modality"):
            get_noise_recipe("cassi", "ultra_dark", photon_db_dict)

    def test_ct_poisson_model(self, photon_db_dict):
        recipe = get_noise_recipe("ct", "standard", photon_db_dict)
        assert recipe["noise_model"] == "poisson"

    def test_mri_gaussian_model(self, photon_db_dict):
        recipe = get_noise_recipe("mri", "standard", photon_db_dict)
        assert recipe["noise_model"] == "gaussian"


# ── PhotonReport integration ────────────────────────────────────────────


class TestPhotonReportLevels:
    """PhotonReport should accept the new optional fields."""

    def _make_base_fields(self):
        """Required fields for a valid PhotonReport."""
        return dict(
            n_photons_per_pixel=1e4,
            snr_db=30.0,
            noise_regime="shot_limited",
            shot_noise_sigma=0.01,
            read_noise_sigma=0.001,
            total_noise_sigma=0.011,
            feasible=True,
            quality_tier="excellent",
            throughput_chain=[{"element": 0.9}],
            noise_model="mixed_poisson_gaussian",
            explanation="Test",
        )

    def test_report_with_recommended_levels(self):
        from pwm_core.agents.contracts import PhotonReport
        fields = self._make_base_fields()
        fields["recommended_levels"] = {
            "bright": {"n_photons": 1e5, "scenario": "Bright"},
            "standard": {"n_photons": 1e4, "scenario": "Standard"},
            "low_light": {"n_photons": 1e3, "scenario": "Low"},
        }
        fields["noise_recipe"] = {
            "noise_model": "poisson_gaussian",
            "default_level": "standard",
        }
        report = PhotonReport(**fields)
        assert report.recommended_levels is not None
        assert len(report.recommended_levels) == 3
        assert report.noise_recipe["noise_model"] == "poisson_gaussian"

    def test_report_without_optional_fields(self):
        from pwm_core.agents.contracts import PhotonReport
        fields = self._make_base_fields()
        report = PhotonReport(**fields)
        assert report.recommended_levels is None
        assert report.noise_recipe is None
