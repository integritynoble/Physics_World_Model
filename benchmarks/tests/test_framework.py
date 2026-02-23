"""Unit tests for the benchmark framework core classes."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

import sys
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "pwm_core"))


class TestMetrics:
    """Tests for benchmarks.framework.metrics."""

    def test_psnr_identical(self):
        from benchmarks.framework.metrics import compute_psnr
        x = np.random.rand(64, 64).astype(np.float32)
        assert compute_psnr(x, x) == 100.0

    def test_psnr_noisy(self):
        from benchmarks.framework.metrics import compute_psnr
        x = np.random.rand(64, 64).astype(np.float32)
        y = x + np.random.randn(64, 64).astype(np.float32) * 0.01
        psnr = compute_psnr(x, y, max_val=1.0)
        assert 30 < psnr < 60

    def test_ssim_identical(self):
        from benchmarks.framework.metrics import compute_ssim
        x = np.random.rand(64, 64).astype(np.float32)
        ssim = compute_ssim(x, x)
        assert ssim > 0.99

    def test_sam_2d_returns_zero(self):
        from benchmarks.framework.metrics import compute_sam
        x = np.random.rand(64, 64).astype(np.float32)
        assert compute_sam(x, x) == 0.0

    def test_sam_3d(self):
        from benchmarks.framework.metrics import compute_sam
        x = np.random.rand(32, 32, 10).astype(np.float32) + 0.1
        sam = compute_sam(x, x)
        assert sam < 1.0  # Should be near zero for identical

    def test_nrmse(self):
        from benchmarks.framework.metrics import compute_nrmse
        x = np.random.rand(64, 64).astype(np.float32)
        assert compute_nrmse(x, x) < 1e-6
        y = x + 0.1
        assert compute_nrmse(x, y) > 0

    def test_metric_set_evaluate(self):
        from benchmarks.framework.metrics import MetricSet
        ms = MetricSet(names=["psnr", "ssim"], primary="psnr", thresholds={"psnr": 20.0})
        x = np.random.rand(32, 32).astype(np.float32)
        y = x + np.random.randn(32, 32).astype(np.float32) * 0.05
        results = ms.evaluate(x, y)
        assert "psnr" in results
        assert "ssim" in results
        assert isinstance(results["psnr"], float)

    def test_metric_set_check_pass(self):
        from benchmarks.framework.metrics import MetricSet
        ms = MetricSet(thresholds={"psnr": 20.0})
        assert ms.check_pass({"psnr": 30.0}) is True
        assert ms.check_pass({"psnr": 15.0}) is False


class TestBenchmarkConfig:
    """Tests for benchmarks.framework.benchmark_config."""

    def test_load_config(self):
        from benchmarks.framework.benchmark_config import load_config
        cfg_path = ROOT / "benchmarks" / "configs" / "widefield.yaml"
        if cfg_path.exists():
            cfg = load_config(cfg_path)
            assert cfg.modality_id == "widefield"
            assert cfg.category == "Microscopy"
            assert cfg.tier in ("A", "B", "C")

    def test_load_all_configs(self):
        from benchmarks.framework.benchmark_config import load_all_configs
        configs = load_all_configs()
        assert len(configs) == 168, f"Expected 168 configs, got {len(configs)}"

    def test_config_has_required_fields(self):
        from benchmarks.framework.benchmark_config import load_all_configs
        configs = load_all_configs()
        for mid, cfg in configs.items():
            assert cfg.modality_id, f"{mid} missing modality_id"
            assert cfg.category, f"{mid} missing category"
            assert cfg.canonical_dag, f"{mid} missing canonical_dag"
            assert cfg.carrier, f"{mid} missing carrier"
            assert len(cfg.x_shape) >= 2, f"{mid} x_shape too short"

    def test_config_from_dict(self):
        from benchmarks.framework.benchmark_config import BenchmarkConfig
        d = {
            "modality_id": "test",
            "display_name": "Test",
            "category": "Test",
            "canonical_dag": "C --> D",
            "carrier": "Photon",
            "x_shape": [64, 64],
            "y_shape": [64, 64],
            "tier": "C",
        }
        cfg = BenchmarkConfig.from_dict(d)
        assert cfg.modality_id == "test"
        assert cfg.dims == (64, 64)


class TestSourceAttribution:
    """Tests for benchmarks.framework.source_attribution."""

    def test_to_dict(self):
        from benchmarks.framework.source_attribution import (
            SourceAttribution, SourceRef, SourceType,
        )
        attr = SourceAttribution(
            ground_truth=SourceRef(SourceType.web, "Test dataset"),
            forward_model=SourceRef(SourceType.registry, "pwm graph"),
        )
        d = attr.to_dict()
        assert d["ground_truth"]["type"] == "web"
        assert d["forward_model"]["reference"] == "pwm graph"


class TestSyntheticData:
    """Tests for benchmarks.framework.synthetic_data."""

    def test_shepp_logan(self):
        from benchmarks.framework.synthetic_data import SheppLoganGenerator
        gen = SheppLoganGenerator(seed=42)
        img = gen.generate((64, 64))
        assert img.shape == (64, 64)
        assert img.min() >= 0.0
        assert img.max() <= 1.0

    def test_cell_phantom(self):
        from benchmarks.framework.synthetic_data import CellPhantomGenerator
        gen = CellPhantomGenerator(seed=42)
        img = gen.generate((64, 64))
        assert img.shape == (64, 64)

    def test_spectral_scene(self):
        from benchmarks.framework.synthetic_data import SpectralSceneGenerator
        gen = SpectralSceneGenerator(seed=42)
        cube = gen.generate((32, 32, 10))
        assert cube.shape == (32, 32, 10)

    def test_generator_registry(self):
        from benchmarks.framework.synthetic_data import get_generator
        gen = get_generator("shepp_logan")
        assert gen is not None


class TestMismatchEngine:
    """Tests for benchmarks.framework.mismatch_engine."""

    def test_generate_m0_scenarios(self):
        from benchmarks.framework.benchmark_config import MismatchParam
        from benchmarks.framework.mismatch_engine import MismatchEngine
        params = [MismatchParam("sigma", 2.0, (1.0, 3.0), "px")]
        engine = MismatchEngine(params)
        scenarios = engine.generate_scenarios("M0")
        assert len(scenarios) == 1
        assert scenarios[0].name == "nominal"

    def test_generate_m1_scenarios(self):
        from benchmarks.framework.benchmark_config import MismatchParam
        from benchmarks.framework.mismatch_engine import MismatchEngine
        params = [
            MismatchParam("sigma", 2.0, (1.0, 3.0), "px"),
            MismatchParam("bg", 50.0, (0, 200), "counts"),
        ]
        engine = MismatchEngine(params)
        scenarios = engine.generate_scenarios("M1")
        # nominal + 2 params * 2 (lo/hi) = 5
        assert len(scenarios) == 5

    def test_generate_m2_scenarios(self):
        from benchmarks.framework.benchmark_config import MismatchParam
        from benchmarks.framework.mismatch_engine import MismatchEngine
        params = [MismatchParam("sigma", 2.0, (1.0, 3.0), "px")]
        engine = MismatchEngine(params)
        scenarios = engine.generate_scenarios("M2")
        assert len(scenarios) > 1


class TestReportWriter:
    """Tests for benchmarks.framework.report_writer."""

    def test_save_json(self):
        from benchmarks.framework.report_writer import ReportWriter, RunBundle
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ReportWriter(Path(tmpdir))
            bundle = RunBundle(
                modality_id="test",
                level="M0",
                metrics={"psnr": 30.0, "ssim": 0.9},
            )
            path = writer.save_json(bundle)
            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert data["modality_id"] == "test"
            assert data["metrics"]["psnr"] == 30.0

    def test_save_markdown(self):
        from benchmarks.framework.report_writer import ReportWriter, RunBundle
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ReportWriter(Path(tmpdir))
            bundle = RunBundle(
                modality_id="test",
                level="M0",
                metrics={"psnr": 30.0},
            )
            path = writer.save_markdown(bundle)
            assert path.exists()
            text = path.read_text()
            assert "test" in text
            assert "30.0" in text


class TestDataSource:
    """Tests for benchmarks.framework.data_source."""

    def test_generate_fallback(self):
        from benchmarks.framework.data_source import DataSource
        ds = DataSource(priority=["generated"], dims=(32, 32))
        result = ds.acquire()
        assert result.x_true.shape == (32, 32)
        assert result.source_type.value == "generated"

    def test_generate_3d(self):
        from benchmarks.framework.data_source import DataSource
        ds = DataSource(priority=["generated"], dims=(32, 32, 10))
        result = ds.acquire()
        assert result.x_true.shape == (32, 32, 10)


class TestCategoryModules:
    """Tests for category modules."""

    def test_microscopy_psf_phantom(self):
        from benchmarks.categories.microscopy_psf import generate_phantom
        img = generate_phantom((64, 64))
        assert img.shape == (64, 64)
        assert img.max() > 0

    def test_microscopy_psf_gaussian(self):
        from benchmarks.categories.microscopy_psf import gaussian_psf
        psf = gaussian_psf(15, sigma=2.0)
        assert psf.shape == (15, 15)
        assert abs(psf.sum() - 1.0) < 1e-5

    def test_ct_radon_phantom(self):
        from benchmarks.categories.medical_ct_radon import generate_phantom
        img = generate_phantom((64, 64))
        assert img.shape == (64, 64)

    def test_mri_phantom(self):
        from benchmarks.categories.medical_mri_kspace import generate_phantom
        img = generate_phantom((64, 64))
        assert img.shape == (64, 64)

    def test_compressive_mask_phantom(self):
        from benchmarks.categories.compressive_mask import generate_phantom
        cube = generate_phantom((32, 32, 10))
        assert cube.shape == (32, 32, 10)

    def test_electron_ctf_phantom(self):
        from benchmarks.categories.electron_ctf import generate_phantom
        img = generate_phantom((64, 64))
        assert img.shape == (64, 64)

    def test_scanning_probe_phantom(self):
        from benchmarks.categories.scanning_probe import generate_phantom
        img = generate_phantom((64, 64))
        assert img.shape == (64, 64)

    def test_remote_sensing_phantom(self):
        from benchmarks.categories.remote_sensing_sar import generate_phantom
        img = generate_phantom((64, 64))
        assert img.shape == (64, 64)

    def test_nuclear_emission_phantom(self):
        from benchmarks.categories.nuclear_emission import generate_phantom
        img = generate_phantom((64, 64))
        assert img.shape == (64, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
