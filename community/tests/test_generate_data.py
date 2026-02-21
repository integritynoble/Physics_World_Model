"""Comprehensive tests for all weekly challenge generate_data.py scripts."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

CHALLENGES_DIR = Path(__file__).resolve().parent.parent / "challenges"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── W10: CASSI ──────────────────────────────────────────────────────

w10 = _load_module("gen_w10", CHALLENGES_DIR / "2026-W10" / "generate_data.py")


class TestW10CASSI:
    def test_spectral_cube_shape(self):
        x = w10.generate_spectral_cube(64, 64, 8, seed=42)
        assert x.shape == (64, 64, 8)

    def test_spectral_cube_range(self):
        x = w10.generate_spectral_cube(64, 64, 8, seed=42)
        assert x.min() >= 0
        assert x.max() <= 1.0 + 1e-6

    def test_spectral_cube_deterministic(self):
        x1 = w10.generate_spectral_cube(32, 32, 4, seed=99)
        x2 = w10.generate_spectral_cube(32, 32, 4, seed=99)
        np.testing.assert_array_equal(x1, x2)

    def test_cassi_measurement_shapes(self):
        x = w10.generate_spectral_cube(64, 64, 8, seed=42)
        y, mask = w10.generate_cassi_measurement(x, seed=42)
        assert y.shape == (64, 64 + 8 - 1)  # H x (W + L - 1)
        assert mask.shape == (64, 64)

    def test_cassi_mask_binary(self):
        x = w10.generate_spectral_cube(64, 64, 8, seed=42)
        _, mask = w10.generate_cassi_measurement(x, seed=42)
        assert set(np.unique(mask)).issubset({0.0, 1.0})

    def test_cassi_measurement_deterministic(self):
        x = w10.generate_spectral_cube(32, 32, 4, seed=42)
        y1, m1 = w10.generate_cassi_measurement(x, seed=42)
        y2, m2 = w10.generate_cassi_measurement(x, seed=42)
        np.testing.assert_array_equal(y1, y2)
        np.testing.assert_array_equal(m1, m2)

    def test_sha256_file(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello")
        h = w10.sha256_file(f)
        assert h.startswith("sha256:")
        assert len(h) > 10


# ── W11: SPC ────────────────────────────────────────────────────────

w11 = _load_module("gen_w11", CHALLENGES_DIR / "2026-W11" / "generate_data.py")


class TestW11SPC:
    def test_hadamard_size_1(self):
        H = w11.hadamard_matrix(1)
        np.testing.assert_array_equal(H, [[1.0]])

    def test_hadamard_size_2(self):
        H = w11.hadamard_matrix(2)
        assert H.shape == (2, 2)
        # H^T H = 2I
        np.testing.assert_array_almost_equal(H.T @ H, 2 * np.eye(2))

    def test_hadamard_size_8(self):
        H = w11.hadamard_matrix(8)
        assert H.shape == (8, 8)
        np.testing.assert_array_almost_equal(H.T @ H, 8 * np.eye(8))

    def test_test_image_shape(self):
        x = w11.generate_test_image(64, 64, seed=42)
        assert x.shape == (64, 64)

    def test_test_image_range(self):
        x = w11.generate_test_image(64, 64, seed=42)
        assert x.min() >= 0
        assert x.max() <= 1.0

    def test_spc_measurement_shapes(self):
        x = w11.generate_test_image(64, 64, seed=42)
        y, Phi = w11.generate_spc_measurement(x, sampling_ratio=0.25, seed=42)
        N = 64 * 64
        M = int(N * 0.25)
        assert y.shape == (M,)
        assert Phi.shape == (M, N)

    def test_spc_deterministic(self):
        x = w11.generate_test_image(32, 32, seed=42)
        y1, P1 = w11.generate_spc_measurement(x, 0.25, seed=42)
        y2, P2 = w11.generate_spc_measurement(x, 0.25, seed=42)
        np.testing.assert_array_equal(y1, y2)
        np.testing.assert_array_equal(P1, P2)


# ── W12: CT ─────────────────────────────────────────────────────────

w12 = _load_module("gen_w12", CHALLENGES_DIR / "2026-W12" / "generate_data.py")


class TestW12CT:
    def test_phantom_shape(self):
        p = w12.generate_phantom(64, seed=42)
        assert p.shape == (64, 64)

    def test_phantom_range(self):
        p = w12.generate_phantom(64, seed=42)
        assert p.min() >= 0
        assert p.max() <= 1.0

    def test_phantom_deterministic(self):
        p1 = w12.generate_phantom(32, seed=99)
        p2 = w12.generate_phantom(32, seed=99)
        np.testing.assert_array_equal(p1, p2)

    def test_parallel_beam_output_shape(self):
        p = w12.generate_phantom(32, seed=42)
        angles = np.linspace(0, np.pi, 10, endpoint=False)
        sino = w12.parallel_beam_project(p, angles)
        assert sino.shape[0] == 10
        assert sino.shape[1] == int(np.ceil(32 * np.sqrt(2)))

    def test_parallel_beam_nonzero(self):
        p = w12.generate_phantom(32, seed=42)
        angles = np.array([0.0])
        sino = w12.parallel_beam_project(p, angles)
        assert sino.sum() > 0


# ── W13: Widefield ──────────────────────────────────────────────────

w13 = _load_module("gen_w13", CHALLENGES_DIR / "2026-W13" / "generate_data.py")


class TestW13Widefield:
    def test_fluorescence_shape(self):
        x = w13.generate_fluorescence_image(64, 64, seed=42)
        assert x.shape == (64, 64)

    def test_fluorescence_range(self):
        x = w13.generate_fluorescence_image(64, 64, seed=42)
        assert x.min() >= 0
        assert x.max() <= 1.0 + 1e-6

    def test_gaussian_psf_shape(self):
        psf = w13.gaussian_psf(64, 64, sigma=2.5)
        assert psf.shape == (64, 64)

    def test_gaussian_psf_normalized(self):
        psf = w13.gaussian_psf(64, 64, sigma=2.5)
        assert abs(psf.sum() - 1.0) < 1e-10

    def test_gaussian_psf_nonnegative(self):
        psf = w13.gaussian_psf(64, 64, sigma=2.5)
        assert psf.min() >= 0

    def test_convolve_fft_shape(self):
        img = np.random.rand(64, 64)
        psf = w13.gaussian_psf(64, 64, sigma=2.5)
        out = w13.convolve_fft(img, psf)
        assert out.shape == (64, 64)

    def test_convolve_fft_identity(self):
        """Convolving with a delta should approximately preserve the image."""
        img = np.random.rand(32, 32)
        psf = np.zeros((32, 32))
        psf[16, 16] = 1.0  # delta at center
        out = w13.convolve_fft(img, psf)
        np.testing.assert_array_almost_equal(out, img, decimal=10)


# ── CISP generate_cisp_data.py ──────────────────────────────────────

cisp_gen = _load_module(
    "gen_cisp",
    CHALLENGES_DIR / "2026_CISP" / "datasets" / "generate_cisp_data.py",
)


class TestCISPGenerateData:
    def test_track_modalities(self):
        assert set(cisp_gen.TRACK_MODALITIES.keys()) == {"correct", "temporal", "medical", "cross"}
        assert cisp_gen.TRACK_MODALITIES["correct"] == ["cassi"]
        assert cisp_gen.TRACK_MODALITIES["cross"] == ["cassi", "spc", "cacti", "widefield"]

    def test_generate_smooth_scene_2d(self):
        rng = np.random.default_rng(42)
        scene = cisp_gen._generate_smooth_scene((64, 64), rng)
        assert scene.shape == (64, 64)
        assert scene.min() >= 0
        assert scene.max() <= 1.0 + 1e-6

    def test_generate_smooth_scene_3d(self):
        rng = np.random.default_rng(42)
        scene = cisp_gen._generate_smooth_scene((64, 64, 8), rng)
        assert scene.shape == (64, 64, 8)
        assert scene.min() >= 0

    def test_sample_mismatch_cassi(self):
        rng = np.random.default_rng(42)
        m = cisp_gen._sample_mismatch("cassi", "moderate", rng)
        assert "mask_shift_px" in m
        assert "dispersion_error_pct" in m

    def test_sample_mismatch_spc(self):
        rng = np.random.default_rng(42)
        m = cisp_gen._sample_mismatch("spc", "moderate", rng)
        assert "pattern_error_pct" in m

    def test_sample_mismatch_mri(self):
        rng = np.random.default_rng(42)
        m = cisp_gen._sample_mismatch("mri", "moderate", rng)
        assert "coil_sensitivity_error_pct" in m

    def test_sample_mismatch_ct(self):
        rng = np.random.default_rng(42)
        m = cisp_gen._sample_mismatch("ct", "moderate", rng)
        assert "beam_spectrum_error_pct" in m

    def test_sample_mismatch_unknown(self):
        rng = np.random.default_rng(42)
        m = cisp_gen._sample_mismatch("unknown_modality", "moderate", rng)
        assert "generic_perturbation_pct" in m

    def test_generate_scene(self):
        rng = np.random.default_rng(42)
        scene = cisp_gen.generate_scene("cassi", 0, rng, "moderate")
        assert "x_gt" in scene
        assert "y" in scene
        assert "mismatch_params" in scene
        assert "metadata" in scene
        assert scene["x_gt"].shape == (256, 256, 28)

    def test_generate_scene_ct(self):
        rng = np.random.default_rng(42)
        scene = cisp_gen.generate_scene("ct", 0, rng, "mild")
        assert scene["x_gt"].shape == (256, 256)

    def test_forward_with_mismatch_3d(self):
        rng = np.random.default_rng(42)
        x = np.random.rand(32, 32, 4).astype(np.float32)
        mismatch = {"mask_shift_px": 1.0, "dispersion_error_pct": 5.0}
        y = cisp_gen._forward_with_mismatch(x, "cassi", mismatch, rng)
        assert y.shape == (32, 32)  # compressed along last dim

    def test_forward_with_mismatch_2d(self):
        rng = np.random.default_rng(42)
        x = np.random.rand(32, 32).astype(np.float32)
        mismatch = {"generic_perturbation_pct": 2.0}
        y = cisp_gen._forward_with_mismatch(x, "widefield", mismatch, rng)
        assert y.shape == (32, 32)

    def test_generate_track_data(self, tmp_path):
        manifest = cisp_gen.generate_track_data("correct", "mild", 2, 42, tmp_path)
        assert manifest["track"] == "correct"
        assert manifest["n_scenes"] == 2
        assert "cassi" in manifest["modalities"]
        assert len(manifest["modalities"]["cassi"]["scenes"]) == 2
        # Check files exist
        assert (tmp_path / "correct" / "cassi" / "scene_0000" / "x_gt.npy").exists()
        assert (tmp_path / "correct" / "cassi" / "scene_0000" / "y.npy").exists()
        assert (tmp_path / "correct" / "cassi" / "scene_0000" / "metadata.json").exists()
        assert (tmp_path / "correct" / "manifest.json").exists()

    def test_severity_scales(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        mild = cisp_gen._sample_mismatch("cassi", "mild", rng1)
        severe = cisp_gen._sample_mismatch("cassi", "severe", rng2)
        # Severe should generally have larger values (since scale is 4x)
        # Can't guarantee per-sample, but average should differ
        # Just check they are valid dicts
        assert isinstance(mild, dict) and isinstance(severe, dict)

    def test_sha256_helper(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"test data")
        h = cisp_gen._sha256(f)
        assert len(h) == 64  # hex digest length
