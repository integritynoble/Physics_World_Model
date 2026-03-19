"""Tests for pwm_core.counterfactual.

Uses synthetic 8x8x4 data for fast CI — no real datasets required.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from pwm_core.counterfactual.schema import (
    CounterfactualPackManifest,
    ExpectedBaseline,
    MismatchConfig,
    NoiseConfig,
    RedTeamCategory,
    RegimeKind,
    ScenarioSpec,
    SplitKind,
)
from pwm_core.counterfactual.base_generator import (
    BaseCounterfactualGenerator,
    add_gaussian_noise,
    add_poisson_gaussian_noise,
    compute_psnr,
    compute_ssim,
    sha256_file,
)
from pwm_core.counterfactual.red_team import (
    RED_TEAM_REGISTRY,
    cassi_gate_flip,
    cassi_oof_config,
    cassi_compute_trap_config,
    spc_gate_flip,
    spc_oof_config,
    spc_compute_trap_config,
    cacti_gate_flip,
    cacti_oof_config,
    cacti_compute_trap_config,
    get_red_team_configs,
)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestSchema:
    """Test Pydantic schema models."""

    def test_split_kind_values(self):
        assert SplitKind.probe.value == "probe"
        assert SplitKind.hidden.value == "hidden"

    def test_regime_kind_values(self):
        assert RegimeKind.nominal.value == "nominal"
        assert RegimeKind.gate_flip.value == "gate_flip"
        assert RegimeKind.oof.value == "oof"
        assert RegimeKind.compute_trap.value == "compute_trap"

    def test_red_team_category_count(self):
        assert len(RedTeamCategory) == 7

    def test_mismatch_config_valid(self):
        mc = MismatchConfig(
            name="mask_dx", value=1.5, unit="px",
            range_min=-3.0, range_max=3.0,
        )
        assert mc.name == "mask_dx"
        assert mc.value == 1.5

    def test_mismatch_config_nan_rejected(self):
        with pytest.raises(ValueError, match="NaN|nan"):
            MismatchConfig(
                name="x", value=float("nan"), unit="px",
                range_min=0, range_max=1,
            )

    def test_mismatch_config_inf_rejected(self):
        with pytest.raises(ValueError, match="inf"):
            MismatchConfig(
                name="x", value=float("inf"), unit="px",
                range_min=0, range_max=1,
            )

    def test_mismatch_config_extra_forbid(self):
        with pytest.raises(Exception):
            MismatchConfig(
                name="x", value=1.0, unit="px",
                range_min=0, range_max=1, extra_field="bad",
            )

    def test_noise_config_gaussian_only(self):
        nc = NoiseConfig(noise_sigma=0.01)
        assert nc.noise_alpha is None
        assert nc.noise_sigma == 0.01

    def test_noise_config_poisson_gaussian(self):
        nc = NoiseConfig(noise_alpha=10000.0, noise_sigma=0.01)
        assert nc.noise_alpha == 10000.0

    def test_scenario_spec_valid(self):
        spec = ScenarioSpec(
            scenario_id="probe_scene01_nominal",
            split=SplitKind.probe,
            regime=RegimeKind.nominal,
            scene_id="scene01",
            noise_config=NoiseConfig(noise_sigma=0.01),
            seed=42,
        )
        assert spec.scenario_id == "probe_scene01_nominal"
        assert spec.mismatch_params == []

    def test_scenario_spec_with_mismatch(self):
        mc = MismatchConfig(
            name="mask_dx", value=1.5, unit="px",
            range_min=-3.0, range_max=3.0,
        )
        spec = ScenarioSpec(
            scenario_id="probe_scene01_single_mask_dx_sev0",
            split=SplitKind.probe,
            regime=RegimeKind.single_param,
            red_team_category=RedTeamCategory.mismatch_escalation,
            scene_id="scene01",
            mismatch_params=[mc],
            noise_config=NoiseConfig(noise_alpha=100000.0, noise_sigma=0.01),
            seed=42,
        )
        assert len(spec.mismatch_params) == 1

    def test_expected_baseline(self):
        eb = ExpectedBaseline(
            solver_id="gap_tv",
            scenario_id="probe_scene01_nominal",
            psnr_db=35.0,
            ssim=0.95,
        )
        assert eb.psnr_db == 35.0

    def test_manifest_valid(self):
        m = CounterfactualPackManifest(
            pack_id="cassi_cfpack_v1",
            modality="cassi",
            seeds={"probe": 2026_02_18, "hidden": 9999_02_18},
            n_scenarios=440,
            regimes=["nominal", "single_param", "compound"],
        )
        assert m.n_scenarios == 440

    def test_manifest_json_roundtrip(self):
        m = CounterfactualPackManifest(
            pack_id="test_pack",
            modality="cassi",
            seeds={"probe": 1, "hidden": 2},
            n_scenarios=1,
        )
        data = json.loads(m.model_dump_json())
        m2 = CounterfactualPackManifest(**data)
        assert m2.pack_id == m.pack_id
        assert m2.n_scenarios == m.n_scenarios


# ---------------------------------------------------------------------------
# Base generator helper tests
# ---------------------------------------------------------------------------


class TestHelpers:
    """Test shared physics helpers."""

    def test_add_poisson_gaussian_noise_shape(self):
        y = np.ones((8, 10), dtype=np.float32)
        rng = np.random.default_rng(42)
        y_noisy = add_poisson_gaussian_noise(y, peak=10000, sigma=0.01, rng=rng)
        assert y_noisy.shape == y.shape
        assert y_noisy.dtype == np.float32

    def test_add_poisson_gaussian_noise_nonneg(self):
        y = np.ones((8, 10), dtype=np.float32) * 0.5
        rng = np.random.default_rng(42)
        y_noisy = add_poisson_gaussian_noise(y, peak=100, sigma=0.1, rng=rng)
        assert np.all(y_noisy >= 0)

    def test_add_gaussian_noise_shape(self):
        y = np.ones((8, 10), dtype=np.float32)
        rng = np.random.default_rng(42)
        y_noisy = add_gaussian_noise(y, sigma=0.01, rng=rng)
        assert y_noisy.shape == y.shape

    def test_compute_psnr_identical(self):
        x = np.ones((8, 8), dtype=np.float32)
        assert compute_psnr(x, x) == 100.0

    def test_compute_psnr_finite(self):
        x = np.ones((8, 8), dtype=np.float32)
        y = x + 0.1 * np.random.randn(8, 8).astype(np.float32)
        psnr = compute_psnr(x, y)
        assert 0 < psnr < 100

    def test_compute_ssim_identical(self):
        x = np.random.rand(8, 8).astype(np.float32)
        ssim = compute_ssim(x, x)
        assert abs(ssim - 1.0) < 1e-6

    def test_sha256_file(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"hello world")
            path = Path(f.name)
        try:
            h = sha256_file(path)
            assert len(h) == 64
            assert all(c in "0123456789abcdef" for c in h)
        finally:
            path.unlink()


# ---------------------------------------------------------------------------
# Red Team tests
# ---------------------------------------------------------------------------


class TestRedTeam:
    """Test red team injection configs."""

    def test_registry_has_all_modalities(self):
        assert set(RED_TEAM_REGISTRY.keys()) == {"cassi", "spc", "cacti"}

    def test_each_modality_has_three_categories(self):
        for mod in ["cassi", "spc", "cacti"]:
            assert set(RED_TEAM_REGISTRY[mod].keys()) == {
                "gate_flip", "oof", "compute_trap"
            }

    def test_cassi_gate_flip_structure(self):
        cfg = cassi_gate_flip()
        assert "mismatch" in cfg
        assert "noise" in cfg
        assert "metadata" in cfg
        assert cfg["noise"]["noise_alpha"] == 500.0

    def test_cassi_oof_has_post_injection(self):
        cfg = cassi_oof_config()
        assert callable(cfg["post_injection"])

    def test_cassi_compute_trap_metadata(self):
        cfg = cassi_compute_trap_config()
        assert cfg["metadata"]["trap_type"] == "polynomial_dispersion"

    def test_spc_gate_flip_high_noise(self):
        cfg = spc_gate_flip()
        assert cfg["noise"]["noise_sigma"] == 0.15

    def test_spc_oof_has_post_injection(self):
        cfg = spc_oof_config()
        assert callable(cfg["post_injection"])

    def test_cacti_gate_flip_low_alpha(self):
        cfg = cacti_gate_flip()
        assert cfg["noise"]["noise_alpha"] == 50.0

    def test_cacti_compute_trap_per_frame(self):
        cfg = cacti_compute_trap_config()
        assert cfg["metadata"]["search_dim"] == 24

    def test_get_red_team_configs_returns_dict(self):
        cfgs = get_red_team_configs("cassi")
        assert isinstance(cfgs, dict)
        assert len(cfgs) == 3

    def test_oof_injection_modifies_measurement(self):
        """Test that oof post-injection actually modifies measurements."""
        cfg = cassi_oof_config()
        fn = cfg["post_injection"]
        rng = np.random.default_rng(42)
        y = np.random.rand(8, 10).astype(np.float32) * 0.5 + 0.1
        y_mod = fn(y, np.zeros((8, 8, 4)), np.ones((8, 8)), rng)
        assert not np.allclose(y, y_mod), "oof injection should modify y"


# ---------------------------------------------------------------------------
# Tiny synthetic generator for testing
# ---------------------------------------------------------------------------


class TinyCounterfactualGenerator(BaseCounterfactualGenerator):
    """Tiny synthetic generator for fast CI tests.

    Uses 8x8x4 data — no real datasets required.
    """

    def __init__(self, n_scenes: int = 2):
        super().__init__(
            modality="tiny",
            pack_id="tiny_cfpack_test",
            seed_public=42,
            seed_hidden=99,
        )
        self.n_scenes = n_scenes

    def load_scenes(self) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        rng = np.random.default_rng(123)
        scenes = []
        for i in range(self.n_scenes):
            x_gt = rng.random((8, 8, 4), dtype=np.float32)
            mask = (rng.random((8, 8), dtype=np.float32) > 0.5).astype(np.float32)
            scenes.append((f"tiny_{i:02d}", x_gt, mask))
        return scenes

    def forward_model(
        self,
        x_gt: np.ndarray,
        mask: np.ndarray,
        mismatch_params: Dict[str, float],
        noise_config: Dict[str, float],
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        H, W, T = x_gt.shape
        scale = mismatch_params.get("scale", 1.0)
        offset = mismatch_params.get("offset", 0.0)

        y = np.sum(x_gt * mask[:, :, np.newaxis], axis=2) * scale + offset
        sigma = noise_config.get("noise_sigma", 0.01)
        y_noisy = add_gaussian_noise(y, sigma=sigma, rng=rng)

        mask_corrupted = mask * scale
        return y_noisy, mask_corrupted

    def get_param_config(self, split: SplitKind) -> Dict[str, Dict[str, float]]:
        if split == SplitKind.probe:
            return {
                "scale":  {"min": 0.9,   "max": 1.1,  "unit": "mult"},
                "offset": {"min": -0.01, "max": 0.01, "unit": "norm"},
            }
        return {
            "scale":  {"min": 0.5,   "max": 1.5,  "unit": "mult"},
            "offset": {"min": -0.1,  "max": 0.1,  "unit": "norm"},
        }

    def get_red_team_configs(self, split: SplitKind) -> Dict[str, Dict[str, Any]]:
        return {
            "gate_flip": {
                "mismatch": {"scale": 1.0},
                "noise": {"noise_sigma": 0.5},
                "metadata": {"red_team": "gate_flip"},
            },
            "oof": {
                "mismatch": {},
                "noise": {"noise_sigma": 0.01},
                "post_injection": lambda y, x, m, rng: y + 0.1 * y**2,
                "metadata": {"red_team": "oof", "effects": ["nonlinear"]},
            },
            "compute_trap": {
                "mismatch": {"scale": 1.05, "offset": 0.005},
                "noise": {"noise_sigma": 0.01},
                "metadata": {"red_team": "compute_trap", "search_dim": 100},
            },
        }

    def get_nominal_noise(self, split: SplitKind) -> Dict[str, float]:
        return {"noise_sigma": 0.01}


# ---------------------------------------------------------------------------
# Generator integration tests
# ---------------------------------------------------------------------------


class TestTinyGenerator:
    """Integration tests with the tiny synthetic generator."""

    def test_generate_pack_structure(self, tmp_path):
        gen = TinyCounterfactualGenerator(n_scenes=2)
        manifest = gen.generate_pack(tmp_path / "tiny_pack")

        assert manifest.modality == "tiny"
        assert manifest.n_scenarios > 0

        # Check both splits exist
        assert (tmp_path / "tiny_pack" / "probe").is_dir()
        assert (tmp_path / "tiny_pack" / "hidden").is_dir()

        # Check manifest file written
        assert (tmp_path / "tiny_pack" / "manifest.json").exists()

    def test_scenario_count(self, tmp_path):
        gen = TinyCounterfactualGenerator(n_scenes=2)
        manifest = gen.generate_pack(tmp_path / "pack")

        # Per scene: 1 nominal + 2 params * 3 sev = 6 single + 3 compound
        #          + 3 red-team = 13 per scene per split
        # 2 scenes * 2 splits * 13 = 52
        assert manifest.n_scenarios == 52

    def test_scenarios_json_written(self, tmp_path):
        gen = TinyCounterfactualGenerator(n_scenes=1)
        gen.generate_pack(tmp_path / "pack")

        probe_scenarios = tmp_path / "pack" / "probe" / "scenarios.json"
        assert probe_scenarios.exists()
        data = json.loads(probe_scenarios.read_text())
        assert isinstance(data, list)
        assert len(data) > 0

    def test_npz_files_created(self, tmp_path):
        gen = TinyCounterfactualGenerator(n_scenes=1)
        gen.generate_pack(tmp_path / "pack")

        # Find all .npz files
        npz_files = list((tmp_path / "pack").rglob("*.npz"))
        assert len(npz_files) > 0

        # Check contents of first npz
        data = np.load(str(npz_files[0]))
        assert "y" in data
        assert "x_gt" in data
        assert "mask_ideal" in data

    def test_probe_has_mask_corrupted(self, tmp_path):
        gen = TinyCounterfactualGenerator(n_scenes=1)
        gen.generate_pack(tmp_path / "pack")

        probe_npz = list((tmp_path / "pack" / "probe").rglob("*.npz"))
        assert len(probe_npz) > 0
        data = np.load(str(probe_npz[0]))
        assert "mask_corrupted" in data

    def test_hidden_no_mask_corrupted(self, tmp_path):
        gen = TinyCounterfactualGenerator(n_scenes=1)
        gen.generate_pack(tmp_path / "pack")

        hidden_npz = list((tmp_path / "pack" / "hidden").rglob("*.npz"))
        assert len(hidden_npz) > 0
        data = np.load(str(hidden_npz[0]))
        assert "mask_corrupted" not in data

    def test_hidden_params_redacted(self, tmp_path):
        gen = TinyCounterfactualGenerator(n_scenes=1)
        gen.generate_pack(tmp_path / "pack")

        hidden_params = list((tmp_path / "pack" / "hidden").rglob("params.json"))
        assert len(hidden_params) > 0
        data = json.loads(hidden_params[0].read_text())
        assert data["mismatch_params"] == "REDACTED"

    def test_probe_params_visible(self, tmp_path):
        gen = TinyCounterfactualGenerator(n_scenes=1)
        gen.generate_pack(tmp_path / "pack")

        probe_params = list((tmp_path / "pack" / "probe").rglob("params.json"))
        assert len(probe_params) > 0
        data = json.loads(probe_params[0].read_text())
        assert isinstance(data["mismatch_params"], dict)

    def test_file_hashes_populated(self, tmp_path):
        gen = TinyCounterfactualGenerator(n_scenes=1)
        manifest = gen.generate_pack(tmp_path / "pack")
        assert len(manifest.file_hashes) > 0
        for h in manifest.file_hashes.values():
            assert len(h) == 64

    def test_regimes_in_manifest(self, tmp_path):
        gen = TinyCounterfactualGenerator(n_scenes=1)
        manifest = gen.generate_pack(tmp_path / "pack")

        expected_regimes = {
            "nominal", "single_param", "compound",
            "gate_flip", "oof", "compute_trap",
        }
        assert set(manifest.regimes) == expected_regimes

    def test_red_team_scenarios_present(self, tmp_path):
        gen = TinyCounterfactualGenerator(n_scenes=1)
        manifest = gen.generate_pack(tmp_path / "pack")

        rt_scenarios = [
            s for s in manifest.scenarios if s.red_team_category is not None
        ]
        categories = {s.red_team_category for s in rt_scenarios}
        assert RedTeamCategory.gate_flip in categories
        assert RedTeamCategory.out_of_family in categories
        assert RedTeamCategory.compute_trap in categories

    def test_reproducibility(self, tmp_path):
        """Same seeds produce identical packs."""
        gen1 = TinyCounterfactualGenerator(n_scenes=1)
        m1 = gen1.generate_pack(tmp_path / "pack1")

        gen2 = TinyCounterfactualGenerator(n_scenes=1)
        m2 = gen2.generate_pack(tmp_path / "pack2")

        # Same number of scenarios
        assert m1.n_scenarios == m2.n_scenarios

        # Same seeds in scenarios
        seeds1 = [s.seed for s in m1.scenarios]
        seeds2 = [s.seed for s in m2.scenarios]
        assert seeds1 == seeds2


# ---------------------------------------------------------------------------
# CASSI / SPC / CACTI generator import tests (no data needed)
# ---------------------------------------------------------------------------


class TestGeneratorImports:
    """Verify all generator classes can be imported."""

    def test_import_cassi(self):
        from pwm_core.counterfactual.cassi_generator import (
            CassiCounterfactualGenerator,
            cassi_forward_parametric,
            warp_affine_2d,
        )
        assert CassiCounterfactualGenerator is not None

    def test_import_spc(self):
        from pwm_core.counterfactual.spc_generator import (
            SpcCounterfactualGenerator,
            make_gain_vector_exp,
        )
        assert SpcCounterfactualGenerator is not None

    def test_import_cacti(self):
        from pwm_core.counterfactual.cacti_generator import (
            CactiCounterfactualGenerator,
            cacti_forward,
            warp_mask_3d,
        )
        assert CactiCounterfactualGenerator is not None


# ---------------------------------------------------------------------------
# CASSI forward model unit tests
# ---------------------------------------------------------------------------


class TestCassiForward:
    """Test CASSI forward model helpers."""

    def test_warp_affine_2d_identity(self):
        from pwm_core.counterfactual.cassi_generator import warp_affine_2d
        mask = np.random.rand(16, 16).astype(np.float32)
        warped = warp_affine_2d(mask, dx=0, dy=0, theta=0)
        np.testing.assert_allclose(warped, mask, atol=1e-5)

    def test_warp_affine_2d_translation(self):
        from pwm_core.counterfactual.cassi_generator import warp_affine_2d
        mask = np.zeros((32, 32), dtype=np.float32)
        mask[14:18, 14:18] = 1.0  # 4x4 block at center
        warped = warp_affine_2d(mask, dx=4.0, dy=0, theta=0)
        # Translation should move content — mask should differ from original
        assert not np.allclose(warped, mask, atol=1e-3), (
            "Translation should modify the mask"
        )

    def test_cassi_forward_parametric_shape(self):
        from pwm_core.counterfactual.cassi_generator import cassi_forward_parametric
        scene = np.random.rand(8, 8, 4).astype(np.float32)
        mask = np.ones((8, 8), dtype=np.float32)
        y = cassi_forward_parametric(scene, mask, a1=2.0, alpha=0.0)
        # W_ext = 8 + 2*(4-1) = 14
        assert y.shape == (8, 14)

    def test_cassi_forward_parametric_nominal(self):
        from pwm_core.counterfactual.cassi_generator import cassi_forward_parametric
        scene = np.ones((4, 4, 2), dtype=np.float32)
        mask = np.ones((4, 4), dtype=np.float32)
        y = cassi_forward_parametric(scene, mask, a1=2.0, alpha=0.0)
        # Band 0: y[:,0:4] += 1, Band 1: y[:,2:6] += 1
        assert y.shape == (4, 6)
        assert y[0, 0] == 1.0  # only band 0
        assert y[0, 2] == 2.0  # band 0 + band 1 overlap
        assert y[0, 4] == 1.0  # only band 1


# ---------------------------------------------------------------------------
# SPC forward model unit tests
# ---------------------------------------------------------------------------


class TestSpcForward:
    """Test SPC forward model helpers."""

    def test_make_gain_vector_exp_unity(self):
        from pwm_core.counterfactual.spc_generator import make_gain_vector_exp
        g = make_gain_vector_exp(10, alpha=0.0)
        np.testing.assert_allclose(g, 1.0)

    def test_make_gain_vector_exp_decay(self):
        from pwm_core.counterfactual.spc_generator import make_gain_vector_exp
        g = make_gain_vector_exp(10, alpha=0.1)
        assert g[0] == 1.0
        assert g[-1] < 1.0
        # Should be monotonically decreasing
        assert np.all(np.diff(g) <= 0)

    def test_img2col_roundtrip(self):
        from pwm_core.counterfactual.spc_generator import img2col, imread_cs, BLOCK_SIZE
        img = np.random.rand(33, 33).astype(np.float32)
        _, _, _, ipad, _, _ = imread_cs(img)
        cols = img2col(ipad)
        assert cols.shape == (33 * 33, 1)
        np.testing.assert_allclose(cols[:, 0], img.reshape(-1))


# ---------------------------------------------------------------------------
# CACTI forward model unit tests
# ---------------------------------------------------------------------------


class TestCactiForward:
    """Test CACTI forward model helpers."""

    def test_warp_mask_3d_identity(self):
        from pwm_core.counterfactual.cacti_generator import warp_mask_3d
        mask = np.random.rand(8, 8, 4).astype(np.float32)
        warped = warp_mask_3d(mask, dx=0, dy=0, theta_deg=0)
        np.testing.assert_allclose(warped, np.clip(mask, 0, 1), atol=1e-5)

    def test_cacti_forward_shape(self):
        from pwm_core.counterfactual.cacti_generator import cacti_forward
        x_gt = np.random.rand(8, 8, 4).astype(np.float32)
        mask = np.ones((8, 8, 4), dtype=np.float32)
        y = cacti_forward(x_gt, mask)
        assert y.shape == (8, 8)

    def test_cacti_forward_sum(self):
        from pwm_core.counterfactual.cacti_generator import cacti_forward
        x_gt = np.ones((4, 4, 2), dtype=np.float32)
        mask = np.ones((4, 4, 2), dtype=np.float32)
        y = cacti_forward(x_gt, mask)
        # Each pixel: sum of 2 frames * 1 * 1 = 2
        np.testing.assert_allclose(y, 2.0)

    def test_cacti_forward_gain_offset(self):
        from pwm_core.counterfactual.cacti_generator import cacti_forward
        x_gt = np.ones((4, 4, 2), dtype=np.float32)
        mask = np.ones((4, 4, 2), dtype=np.float32)
        y = cacti_forward(x_gt, mask, gain=2.0, offset=0.5)
        # gain * sum + offset = 2 * 2 + 0.5 = 4.5
        np.testing.assert_allclose(y, 4.5)
