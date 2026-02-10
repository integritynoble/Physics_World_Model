"""test_sharepack.py

Tests for pwm_core.export.sharepack module.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest


def test_generate_teaser_image_returns_pil_image():
    """generate_teaser_image should return a PIL Image."""
    from pwm_core.export.sharepack import generate_teaser_image

    x_gt = np.random.rand(64, 64).astype(np.float32)
    y = np.random.rand(64, 64).astype(np.float32)
    x_hat = np.random.rand(64, 64).astype(np.float32)

    img = generate_teaser_image(x_gt, y, x_hat, modality="test")

    from PIL import Image
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size[0] > 0 and img.size[1] > 0


def test_generate_teaser_image_3d_input():
    """generate_teaser_image should handle 3D arrays (e.g., spectral cubes)."""
    from pwm_core.export.sharepack import generate_teaser_image

    x_gt = np.random.rand(8, 64, 64).astype(np.float32)
    y = np.random.rand(8, 64, 64).astype(np.float32)
    x_hat = np.random.rand(8, 64, 64).astype(np.float32)

    img = generate_teaser_image(x_gt, y, x_hat, modality="cassi")

    from PIL import Image
    assert isinstance(img, Image.Image)


def test_generate_summary_returns_markdown():
    """generate_summary should return markdown with 3 bullets."""
    from pwm_core.export.sharepack import generate_summary

    metrics = {"psnr_db": 30.5, "ssim": 0.92, "runtime_s": 12.3}
    md = generate_summary(metrics, "cassi")

    assert isinstance(md, str)
    assert md.startswith("# PWM")
    assert "**Problem:**" in md
    assert "**Approach:**" in md
    assert "**Result:**" in md
    assert "30.50 dB" in md
    assert "0.920" in md


def test_generate_summary_with_mismatch():
    """generate_summary should include calibration gain when mismatch_info provided."""
    from pwm_core.export.sharepack import generate_summary

    metrics = {"psnr_db": 25.0, "ssim": 0.85, "runtime_s": 5.0}
    mismatch = {"psnr_gain_db": 10.15}
    md = generate_summary(metrics, "cassi", mismatch_info=mismatch)

    assert "calibration gain" in md
    assert "+10.2 dB" in md


def test_generate_metrics_json_filters_keys():
    """generate_metrics_json should only include allowed keys."""
    from pwm_core.export.sharepack import generate_metrics_json

    metrics = {
        "psnr_db": 30.5,
        "ssim": 0.92,
        "runtime_s": 12.3,
        "internal_debug": "should_be_filtered",
        "solver_id": "gap_tv",
    }
    result = generate_metrics_json(metrics)

    assert "psnr_db" in result
    assert "ssim" in result
    assert "runtime_s" in result
    assert "solver_id" in result
    assert "internal_debug" not in result


def test_generate_metrics_json_rejects_nan():
    """generate_metrics_json should skip NaN and Inf values."""
    from pwm_core.export.sharepack import generate_metrics_json

    metrics = {
        "psnr_db": float("nan"),
        "ssim": float("inf"),
        "runtime_s": 5.0,
    }
    result = generate_metrics_json(metrics)

    assert "psnr_db" not in result
    assert "ssim" not in result
    assert "runtime_s" in result


def test_generate_reproduce_sh():
    """generate_reproduce_sh should produce valid bash script content."""
    from pwm_core.export.sharepack import generate_reproduce_sh

    sh = generate_reproduce_sh("cassi", preset="tissue")
    assert "#!/usr/bin/env bash" in sh
    assert "pwm demo cassi --preset tissue" in sh
    assert "--export-sharepack" in sh


def test_export_sharepack_creates_expected_files():
    """export_sharepack should create summary.md, metrics.json, reproduce.sh."""
    from pwm_core.export.sharepack import export_sharepack

    with tempfile.TemporaryDirectory() as tmpdir:
        rb_dir = Path(tmpdir) / "runbundle"
        rb_dir.mkdir()
        data_dir = rb_dir / "data"
        data_dir.mkdir()
        results_dir = rb_dir / "results"
        results_dir.mkdir()

        # Create test arrays
        x_gt = np.random.rand(32, 32).astype(np.float32)
        y = np.random.rand(32, 32).astype(np.float32)
        x_hat = np.random.rand(32, 32).astype(np.float32)
        np.save(str(data_dir / "x_gt.npy"), x_gt)
        np.save(str(data_dir / "y.npy"), y)
        np.save(str(results_dir / "x_hat.npy"), x_hat)

        # Create manifest
        manifest = {
            "version": "0.3.0",
            "spec_id": "test_run",
            "timestamp": "2026-02-09T00:00:00Z",
            "modality": "test_modality",
            "provenance": {"git_hash": "abc1234", "seeds": [42], "platform": "test", "pwm_version": "0.3.0"},
            "metrics": {"psnr_db": 30.0, "ssim": 0.90, "runtime_s": 1.0},
            "artifacts": {"x_gt": "data/x_gt.npy", "y": "data/y.npy", "x_hat": "results/x_hat.npy"},
            "hashes": {"x_gt": "sha256:abc", "y": "sha256:def", "x_hat": "sha256:ghi"},
        }
        (rb_dir / "runbundle_manifest.json").write_text(json.dumps(manifest))

        out_dir = Path(tmpdir) / "sharepack"
        result = export_sharepack(str(rb_dir), str(out_dir), modality="test_modality")

        assert result == out_dir
        assert (out_dir / "summary.md").exists()
        assert (out_dir / "metrics.json").exists()
        assert (out_dir / "reproduce.sh").exists()

        # Verify metrics.json content
        metrics_data = json.loads((out_dir / "metrics.json").read_text())
        assert metrics_data["psnr_db"] == 30.0

        # Verify summary.md content
        summary = (out_dir / "summary.md").read_text()
        assert "TEST MODALITY" in summary
