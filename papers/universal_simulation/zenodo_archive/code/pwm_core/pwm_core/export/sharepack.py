"""pwm_core.export.sharepack

One-command shareable artifact generator for PWM RunBundles.
Produces a sharepack/ directory with teaser.png, teaser.mp4 (or GIF fallback),
summary.md, metrics.json, and reproduce.sh.
"""
from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _to_2d(arr):
    """Extract a representative 2D slice from an N-D array."""
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[arr.shape[0] // 2]
    if arr.ndim >= 4:
        while arr.ndim > 2:
            arr = arr[arr.shape[0] // 2]
        return arr
    side = max(1, int(np.sqrt(arr.size)))
    return arr.flat[: side * side].reshape(side, side)


def _normalize_u8(arr):
    """Normalize array to uint8 [0, 255]."""
    arr = arr.astype(np.float64)
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr - lo) / (hi - lo) * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _draw_simple_label(img, x, y, w, h, text):
    """Draw a text label using PIL ImageDraw."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
    except Exception:
        font = None
    bbox = (
        draw.textbbox((0, 0), text, font=font)
        if font
        else (0, 0, len(text) * 6, 10)
    )
    tw = bbox[2] - bbox[0]
    tx = x + max(0, (w - tw) // 2)
    ty = y + 2
    draw.text((tx, ty), text, fill=(0, 0, 0), font=font)


def generate_teaser_image(x_gt, y, x_hat, modality="unknown"):
    """Create a side-by-side PNG: ground truth | measurement | reconstruction.

    Handles arbitrary-dimensional arrays by selecting the central 2D slice.
    Returns a PIL Image (RGB).
    """
    from PIL import Image

    gt_2d = _normalize_u8(_to_2d(x_gt))
    y_2d = _normalize_u8(_to_2d(y))
    hat_2d = _normalize_u8(_to_2d(x_hat))

    target_h = max(gt_2d.shape[0], y_2d.shape[0], hat_2d.shape[0], 64)
    target_w = max(gt_2d.shape[1], y_2d.shape[1], hat_2d.shape[1], 64)

    def _resize_panel(arr):
        img = Image.fromarray(arr, mode="L")
        return img.resize((target_w, target_h), Image.NEAREST)

    panels = [_resize_panel(gt_2d), _resize_panel(y_2d), _resize_panel(hat_2d)]
    labels = ["Ground Truth", "Measurement", "Reconstruction"]
    gap = 4
    total_w = target_w * 3 + gap * 2
    canvas = Image.new("RGB", (total_w, target_h + 20), color=(255, 255, 255))

    x_offset = 0
    for panel, label in zip(panels, labels):
        rgb_panel = panel.convert("RGB")
        canvas.paste(rgb_panel, (x_offset, 20))
        _draw_simple_label(canvas, x_offset, 0, target_w, 18, label)
        x_offset += target_w + gap

    return canvas


def generate_teaser_video(x_gt, x_hat, duration=30, output_path=None):
    """Generate an animated comparison video/GIF.

    Fallback chain:
    1. mp4 via matplotlib FFMpegWriter
    2. GIF via matplotlib PillowWriter
    3. GIF via pure Pillow
    4. Print message and return None (no error raised)
    """
    if output_path is None:
        output_path = Path(tempfile.mkdtemp()) / "teaser.mp4"
    output_path = Path(output_path)

    gt_2d = _to_2d(x_gt).astype(np.float64)
    hat_2d = _to_2d(x_hat).astype(np.float64)

    def _norm(arr):
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        if hi - lo > 1e-12:
            return (arr - lo) / (hi - lo)
        return np.zeros_like(arr)

    gt_norm = _norm(gt_2d)
    hat_norm = _norm(hat_2d)
    n_frames = max(duration, 10)

    def _blend_frame(t):
        alpha = 0.5 * (1.0 + np.cos(2.0 * np.pi * t / n_frames))
        return alpha * gt_norm + (1.0 - alpha) * hat_norm

    # Strategy 1: matplotlib animation
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.axis("off")
        im = ax.imshow(_blend_frame(0), cmap="gray", vmin=0, vmax=1)
        ax.set_title("Ground Truth <-> Reconstruction")

        def _update(frame):
            im.set_data(_blend_frame(frame))
            return [im]

        anim = animation.FuncAnimation(fig, _update, frames=n_frames, blit=True)

        try:
            writer = animation.FFMpegWriter(fps=10)
            mp4_path = output_path.with_suffix(".mp4")
            anim.save(str(mp4_path), writer=writer)
            plt.close(fig)
            return mp4_path
        except Exception:
            pass

        try:
            gif_path = output_path.with_suffix(".gif")
            writer_gif = animation.PillowWriter(fps=10)
            anim.save(str(gif_path), writer=writer_gif)
            plt.close(fig)
            return gif_path
        except Exception:
            plt.close(fig)
    except ImportError:
        pass

    # Strategy 2: pure Pillow GIF
    try:
        from PIL import Image
        frames = []
        for t in range(min(n_frames, 30)):
            blended = _blend_frame(t)
            blended_u8 = np.clip(blended * 255, 0, 255).astype(np.uint8)
            frames.append(Image.fromarray(blended_u8, mode="L"))
        gif_path = output_path.with_suffix(".gif")
        frames[0].save(
            str(gif_path), save_all=True, append_images=frames[1:],
            duration=100, loop=0,
        )
        return gif_path
    except Exception as exc:
        logger.warning("mp4 skipped; encoder unavailable (%s)", exc)
        print("mp4 skipped; encoder unavailable", file=sys.stderr)
        return None


def generate_summary(metrics, modality, mismatch_info=None):
    """Generate a 3-bullet markdown summary."""
    psnr = metrics.get("psnr_db", "N/A")
    ssim = metrics.get("ssim", "N/A")
    runtime = metrics.get("runtime_s", "N/A")
    modality_display = modality.upper().replace("_", " ")

    bullet1 = (
        "- **Problem:** " + modality_display
        + " imaging -- recover the scene from compressed/encoded measurements."
    )
    bullet2 = (
        "- **Approach:** PWM physics-aware reconstruction pipeline "
        "with operator calibration and solver portfolio."
    )

    result_parts = []
    if isinstance(psnr, (int, float)):
        result_parts.append("PSNR = {:.2f} dB".format(psnr))
    if isinstance(ssim, (int, float)):
        result_parts.append("SSIM = {:.3f}".format(ssim))
    if isinstance(runtime, (int, float)):
        result_parts.append("runtime = {:.1f}s".format(runtime))
    if mismatch_info:
        correction = mismatch_info.get("psnr_gain_db")
        if correction is not None:
            result_parts.append("calibration gain = +{:.1f} dB".format(correction))

    result_str = ", ".join(result_parts) if result_parts else "see metrics.json"
    bullet3 = "- **Result:** " + result_str + "."

    return (
        "# PWM " + modality_display + " Demo\n\n"
        + bullet1 + "\n" + bullet2 + "\n" + bullet3 + "\n"
    )


def generate_metrics_json(metrics):
    """Extract a clean metrics subset suitable for SharePack."""
    allowed_keys = {
        "psnr_db", "ssim", "runtime_s", "sam", "theta_error_rmse",
        "psnr_gain_db", "n_iters", "solver_id",
    }
    out = {}
    for k, v in metrics.items():
        if k in allowed_keys:
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                continue
            out[k] = v
    return out


def generate_reproduce_sh(modality, preset=None):
    """Generate shell script content to reproduce the demo run."""
    preset_flag = " --preset " + preset if preset else ""
    preset_display = " (" + preset + ")" if preset else ""
    lines = [
        "#!/usr/bin/env bash",
        "# Reproduce this PWM demo result",
        "# Generated by pwm export-sharepack",
        "",
        "set -euo pipefail",
        "",
        'echo "=== PWM Demo: ' + modality + preset_display + ' ==="',
        'echo "Installing PWM (if needed)..."',
        "pip install -q pwm-core 2>/dev/null || true",
        "",
        'echo "Running demo..."',
        "pwm demo " + modality + preset_flag + " --run --export-sharepack",
        "",
        'echo "Done! Check sharepack/ for results."',
    ]
    return "\n".join(lines) + "\n"


def export_sharepack(runbundle_dir, output_dir, modality=None, preset=None):
    """Read a RunBundle and generate all SharePack artifacts.

    Parameters
    ----------
    runbundle_dir : str or Path
        Path to RunBundle directory (must contain runbundle_manifest.json).
    output_dir : str or Path
        Output directory for the sharepack.
    modality : str or None
        Modality override. If None, read from manifest.
    preset : str or None
        Preset name for reproduce.sh.

    Returns
    -------
    Path
        Path to the output sharepack directory.
    """
    rb = Path(runbundle_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = rb / "runbundle_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "RunBundle manifest not found: " + str(manifest_path)
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    metrics = manifest.get("metrics", {})
    if modality is None:
        modality = manifest.get("modality", "unknown")

    artifacts = manifest.get("artifacts", {})

    def _load_array(key):
        rel = artifacts.get(key)
        if rel is None:
            return None
        p = rb / rel
        if p.exists():
            return np.load(str(p))
        return None

    x_gt = _load_array("x_gt")
    y_meas = _load_array("y")
    x_hat = _load_array("x_hat")

    # 1) Teaser image
    if x_gt is not None and y_meas is not None and x_hat is not None:
        try:
            teaser_img = generate_teaser_image(x_gt, y_meas, x_hat, modality)
            teaser_path = out / "teaser.png"
            teaser_img.save(str(teaser_path))
        except Exception as exc:
            logger.warning("Teaser image generation failed: %s", exc)

    # 2) Teaser video
    if x_gt is not None and x_hat is not None:
        generate_teaser_video(
            x_gt, x_hat, duration=30, output_path=out / "teaser.mp4"
        )

    # 3) Summary
    mismatch_info = manifest.get("mismatch", None)
    summary_md = generate_summary(metrics, modality, mismatch_info)
    (out / "summary.md").write_text(summary_md, encoding="utf-8")

    # 4) Metrics JSON
    metrics_subset = generate_metrics_json(metrics)
    (out / "metrics.json").write_text(
        json.dumps(metrics_subset, indent=2), encoding="utf-8"
    )

    # 5) Reproduce script
    reproduce_content = generate_reproduce_sh(modality, preset)
    reproduce_path = out / "reproduce.sh"
    reproduce_path.write_text(reproduce_content, encoding="utf-8")
    reproduce_path.chmod(0o755)

    logger.info("SharePack exported to: %s", out)
    return out
