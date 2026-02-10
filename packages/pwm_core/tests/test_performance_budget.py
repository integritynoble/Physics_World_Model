"""test_performance_budget.py

Placeholder performance budget tests for the demo concept.
Verifies that basic demo operations complete within acceptable time bounds.
"""
from __future__ import annotations

import time

import numpy as np
import pytest


def test_teaser_image_generation_under_5s():
    """Teaser image generation should complete in < 5 seconds for 64x64."""
    from pwm_core.export.sharepack import generate_teaser_image

    x_gt = np.random.rand(64, 64).astype(np.float32)
    y = np.random.rand(64, 64).astype(np.float32)
    x_hat = np.random.rand(64, 64).astype(np.float32)

    t0 = time.monotonic()
    img = generate_teaser_image(x_gt, y, x_hat, modality="test")
    elapsed = time.monotonic() - t0

    assert elapsed < 5.0, "Teaser image generation took {:.2f}s (budget: 5s)".format(elapsed)
    assert img is not None


def test_summary_generation_under_1s():
    """Summary generation should be near-instant."""
    from pwm_core.export.sharepack import generate_summary

    metrics = {"psnr_db": 30.0, "ssim": 0.90, "runtime_s": 1.0}

    t0 = time.monotonic()
    md = generate_summary(metrics, "cassi")
    elapsed = time.monotonic() - t0

    assert elapsed < 1.0, "Summary generation took {:.2f}s (budget: 1s)".format(elapsed)
    assert len(md) > 0


def test_metrics_json_generation_under_1s():
    """Metrics JSON extraction should be near-instant."""
    from pwm_core.export.sharepack import generate_metrics_json

    metrics = {"psnr_db": 30.0, "ssim": 0.90, "runtime_s": 1.0, "extra": "ignore"}

    t0 = time.monotonic()
    result = generate_metrics_json(metrics)
    elapsed = time.monotonic() - t0

    assert elapsed < 1.0, "Metrics JSON generation took {:.2f}s (budget: 1s)".format(elapsed)
    assert "psnr_db" in result


def test_casepack_loading_under_2s():
    """CasePack loading should complete within 2 seconds."""
    from pwm_core.cli.demo import _load_casepack

    t0 = time.monotonic()
    cp = _load_casepack("cassi")
    elapsed = time.monotonic() - t0

    assert elapsed < 2.0, "CasePack loading took {:.2f}s (budget: 2s)".format(elapsed)
    assert cp is not None


def test_gallery_data_completeness():
    """Gallery should have data for all 26 modalities."""
    from docs.gallery.generate_gallery import BENCHMARK_DATA

    assert len(BENCHMARK_DATA) == 26, "Expected 26 modalities, got {}".format(len(BENCHMARK_DATA))

    keys = {e["key"] for e in BENCHMARK_DATA}
    expected = {
        "widefield", "widefield_lowdose", "confocal_livecell", "confocal_3d",
        "sim", "cassi", "spc", "cacti", "lensless", "lightsheet", "ct", "mri",
        "ptychography", "holography", "nerf", "gaussian_splatting", "matrix",
        "panorama_multifocal", "light_field", "integral", "phase_retrieval",
        "flim", "photoacoustic", "oct", "fpm", "dot",
    }
    assert keys == expected, "Missing modalities: {}".format(expected - keys)
