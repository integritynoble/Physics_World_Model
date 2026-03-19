#!/usr/bin/env python3
"""MRI Multi-Algorithm Benchmark across public/dev/hidden phantom tiers.

Runs 6 reconstruction algorithms on multi-coil MRI data stored in HDF5 files
produced by the PWM synthetic MRI benchmark generator. Each file contains
multiple samples with 15-coil k-space, 1D phase-encode undersampling masks
(R~4), ground truth images, coil sensitivity maps, B0 maps, and warp fields.

Algorithms:
  1. Zero-filled RSS       — baseline inverse FFT + root-sum-of-squares
  2. SENSE                 — CG-SENSE with ESPIRiT-style sensitivity maps
  3. CS-MRI (wavelet)      — l1-wavelet compressed sensing (auto-delegates to
                             SENSE for multi-coil input)
  4. PnP-HQS              — Plug-and-Play Half-Quadratic Splitting with best
                             available denoiser (DRUNet > BM3D > NLM > Gaussian)
  5. VarNet                — End-to-End Variational Network (single-coil;
                             uses random init when no pretrained weights found)
  6. MoDL                  — Model-Based Deep Learning (single-coil;
                             uses random init when no pretrained weights found)

Notes:
  - VarNet and MoDL run with random weights (no pretrained checkpoint) so their
    results will be near or below the zero-filled baseline.
  - PnP denoiser cascade: DRUNet > BM3D > NLM > Gaussian (uses best available).
  - CS-MRI with multi-coil kspace auto-delegates to SENSE internally.
  - The mask is 1D (320,) representing phase-encode undersampling.

Usage:
    python run_mri_multiphantom.py
    python run_mri_multiphantom.py --tier public --solver sense --max-samples 5
    python run_mri_multiphantom.py --tier all --solver all --device cpu
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[3]  # Physics_World_Model root
# _ROOT must come BEFORE packages/pwm_core because pwm_core also has a
# benchmarks/ subpackage that would shadow the top-level benchmarks/.
sys.path.insert(0, str(_ROOT / "packages" / "pwm_core"))
sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Solver imports (lazy — guarded in dispatcher)
# ---------------------------------------------------------------------------
from pwm_core.recon.mri_solvers import (  # noqa: E402
    zero_filled_reconstruction,
    sense_reconstruction,
    cs_mri_wavelet,
    estimate_sensitivity_maps,
)
from benchmarks.framework.metrics import compute_psnr, compute_ssim  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress noisy warnings from deep learning modules when weights are missing
warnings.filterwarnings("ignore", message=".*weights.*not found.*")
warnings.filterwarnings("ignore", message=".*random.*init.*")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TIERS = {
    "public": _ROOT / "datasets" / "benchmark" / "mri" / "public" / "mri_challenge_public.h5",
    "dev": _ROOT / "datasets" / "benchmark" / "mri" / "dev" / "mri_challenge_dev.h5",
    "hidden": _ROOT / "datasets" / "benchmark" / "mri" / "hidden" / "mri_challenge_hidden.h5",
}

RESULTS_DIR = _ROOT / "papers" / "pwm_flagship" / "results" / "real_data_4scenario"

SOLVER_NAMES = ["zerofilled_rss", "sense", "cs_mri", "pnp_hqs", "varnet", "modl"]

# Map CLI shorthand to internal solver name
SOLVER_CLI_MAP = {
    "zerofilled": "zerofilled_rss",
    "sense": "sense",
    "cs": "cs_mri",
    "pnp": "pnp_hqs",
    "varnet": "varnet",
    "modl": "modl",
}


# ===========================================================================
# Multi-coil MRI forward/adjoint operator for PnP
# ===========================================================================
class MultiCoilMRIOp:
    """Multi-coil MRI forward/adjoint operator for PnP solvers.

    Wraps coil sensitivity maps and a 1D phase-encode undersampling mask
    into callable forward and adjoint operators suitable for pnp_hqs.

    Parameters
    ----------
    coil_maps : np.ndarray
        Complex coil sensitivity maps of shape ``(C, H, W)``.
    mask_1d : np.ndarray
        1D undersampling mask of shape ``(H,)`` (phase-encode direction)
        with dtype uint8 or bool.
    """

    def __init__(self, coil_maps: np.ndarray, mask_1d: np.ndarray):
        self.coil_maps = coil_maps.astype(np.complex64)
        C, H, W = coil_maps.shape
        # Expand 1D mask (phase-encode) to 2D: mask_2d[ky, kx] = mask_1d[ky]
        self.mask_2d = (
            mask_1d.astype(np.float32).reshape(-1, 1)
            * np.ones((1, W), dtype=np.float32)
        )
        self._H = H
        self._W = W
        self._C = C

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward operator: image -> undersampled multi-coil k-space.

        Parameters
        ----------
        x : np.ndarray
            Image of shape ``(H, W)`` (float32).

        Returns
        -------
        np.ndarray
            Multi-coil k-space ``(C, H, W)`` complex64.
        """
        from scipy.fft import fft2, fftshift

        y = np.zeros((self._C, self._H, self._W), dtype=np.complex64)
        x_c = x.astype(np.complex64)
        for c in range(self._C):
            img_c = self.coil_maps[c] * x_c
            y[c] = self.mask_2d * fftshift(fft2(np.fft.ifftshift(img_c), axes=(-2, -1)))
        return y

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Adjoint operator: multi-coil k-space -> image.

        Parameters
        ----------
        y : np.ndarray
            Multi-coil k-space ``(C, H, W)`` complex64.

        Returns
        -------
        np.ndarray
            Image ``(H, W)`` float32 (real part).
        """
        from scipy.fft import ifft2, ifftshift

        x = np.zeros((self._H, self._W), dtype=np.complex64)
        for c in range(y.shape[0]):
            k_masked = self.mask_2d * y[c]
            img_c = ifft2(np.fft.ifftshift(k_masked), axes=(-2, -1))
            x += np.conj(self.coil_maps[c]) * img_c
        return np.real(x).astype(np.float32)


# ===========================================================================
# Coil combination for single-coil solvers (VarNet, MoDL)
# ===========================================================================
def coil_combine_rss(y_kspace: np.ndarray) -> np.ndarray:
    """Combine multi-coil k-space to single-coil via RSS in image domain.

    Parameters
    ----------
    y_kspace : np.ndarray
        Multi-coil k-space ``(C, H, W)`` complex.

    Returns
    -------
    np.ndarray
        Single-coil k-space ``(H, W)`` complex128.
    """
    from scipy.fft import ifft2

    imgs = ifft2(np.fft.ifftshift(y_kspace, axes=(-2, -1)), axes=(-2, -1))
    rss = np.sqrt(np.sum(np.abs(imgs) ** 2, axis=0))
    kspace_combined = np.fft.fftshift(np.fft.fft2(rss))
    return kspace_combined


# ===========================================================================
# Normalisation helper
# ===========================================================================
def _normalise_to_01(x: np.ndarray) -> np.ndarray:
    """Take magnitude (if complex) and normalise to [0, 1]."""
    if np.iscomplexobj(x):
        x = np.abs(x)
    x = x.astype(np.float64)
    x_max = x.max()
    if x_max > 1e-10:
        x = x / x_max
    return x.astype(np.float32)


# ===========================================================================
# Solver dispatcher
# ===========================================================================
def _expand_mask_2d(mask_1d: np.ndarray, H: int, W: int) -> np.ndarray:
    """Expand 1D phase-encode mask (H,) to 2D (H, W) float32."""
    return (
        mask_1d.astype(np.float32).reshape(-1, 1)
        * np.ones((1, W), dtype=np.float32)
    )


def solve_zerofilled(
    x_true: np.ndarray,
    y_kspace: np.ndarray,
    mask: np.ndarray,
    coil_maps: np.ndarray,
    device: Optional[str],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Zero-filled RSS reconstruction."""
    x_hat = zero_filled_reconstruction(y_kspace, mask=None, device=device)
    x_hat = _normalise_to_01(x_hat)
    return x_hat, {"solver": "zerofilled_rss"}


def solve_sense(
    x_true: np.ndarray,
    y_kspace: np.ndarray,
    mask: np.ndarray,
    coil_maps: np.ndarray,
    device: Optional[str],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """SENSE reconstruction using provided coil maps."""
    H, W = y_kspace.shape[1], y_kspace.shape[2]
    mask_2d = _expand_mask_2d(mask, H, W)
    x_hat = sense_reconstruction(
        y_kspace,
        sensitivity_maps=coil_maps,
        mask=mask_2d,
        regularization=0.001,
        iterations=30,
        device=device,
    )
    x_hat = _normalise_to_01(x_hat)
    return x_hat, {"solver": "sense", "iterations": 30, "regularization": 0.001}


def solve_cs_mri(
    x_true: np.ndarray,
    y_kspace: np.ndarray,
    mask: np.ndarray,
    coil_maps: np.ndarray,
    device: Optional[str],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """CS-MRI wavelet reconstruction (multi-coil auto-delegates to SENSE)."""
    H, W = y_kspace.shape[1], y_kspace.shape[2]
    mask_2d = _expand_mask_2d(mask, H, W)
    x_hat = cs_mri_wavelet(
        y_kspace,
        mask=mask_2d,
        lam=0.01,
        iterations=50,
        sensitivity_maps=coil_maps,
        device=device,
    )
    x_hat = _normalise_to_01(x_hat)
    return x_hat, {"solver": "cs_mri", "lam": 0.01, "iterations": 50}


def solve_pnp_hqs(
    x_true: np.ndarray,
    y_kspace: np.ndarray,
    mask: np.ndarray,
    coil_maps: np.ndarray,
    device: Optional[str],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """PnP-HQS reconstruction with best available denoiser."""
    from pwm_core.recon.pnp import pnp_hqs, get_denoiser

    dev_str = device if device is not None else "cpu"
    denoiser = get_denoiser(denoiser_type="auto", device=dev_str)

    mri_op = MultiCoilMRIOp(coil_maps, mask)

    H, W = y_kspace.shape[1], y_kspace.shape[2]
    x_hat = pnp_hqs(
        y=y_kspace,
        forward=mri_op.forward,
        adjoint=mri_op.adjoint,
        x_shape=(H, W),
        denoiser=denoiser,
        iters=30,
        rho=1.0,
        sigma=0.1,
        sigma_decay=0.9,
    )
    x_hat = _normalise_to_01(x_hat)

    denoiser_name = type(denoiser).__name__ if hasattr(denoiser, "__name__") is False else denoiser.__name__
    return x_hat, {"solver": "pnp_hqs", "denoiser": str(denoiser_name), "iters": 30}


def solve_varnet(
    x_true: np.ndarray,
    y_kspace: np.ndarray,
    mask: np.ndarray,
    coil_maps: np.ndarray,
    device: Optional[str],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """VarNet reconstruction (single-coil; random init without weights)."""
    from pwm_core.recon.varnet import varnet_recon

    kspace_sc = coil_combine_rss(y_kspace).astype(np.complex64)
    # VarNet expects (H,W) complex and (H,W) or (W,) binary mask
    x_hat = varnet_recon(
        kspace_sc,
        mask=mask,
        weights_path=None,
        n_cascades=12,
        device=device,
    )
    x_hat = _normalise_to_01(x_hat)
    return x_hat, {"solver": "varnet", "n_cascades": 12, "note": "random_init_no_pretrained_weights"}


def solve_modl(
    x_true: np.ndarray,
    y_kspace: np.ndarray,
    mask: np.ndarray,
    coil_maps: np.ndarray,
    device: Optional[str],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """MoDL reconstruction (single-coil; random init without weights)."""
    from pwm_core.recon.modl import modl_recon

    kspace_sc = coil_combine_rss(y_kspace).astype(np.complex64)
    x_hat = modl_recon(
        kspace_sc,
        mask=mask,
        weights_path=None,
        n_iter=5,
        device=device,
    )
    x_hat = _normalise_to_01(x_hat)
    return x_hat, {"solver": "modl", "n_iter": 5, "note": "random_init_no_pretrained_weights"}


# Dispatcher mapping
SOLVER_DISPATCH: Dict[str, Callable] = {
    "zerofilled_rss": solve_zerofilled,
    "sense": solve_sense,
    "cs_mri": solve_cs_mri,
    "pnp_hqs": solve_pnp_hqs,
    "varnet": solve_varnet,
    "modl": solve_modl,
}


# ===========================================================================
# Sample loader
# ===========================================================================
def load_samples(h5_path: Path, max_samples: int = 0) -> List[Dict[str, Any]]:
    """Load all samples from an HDF5 benchmark file.

    Parameters
    ----------
    h5_path : Path
        Path to the HDF5 file.
    max_samples : int
        Maximum number of samples to load (0 = all).

    Returns
    -------
    list of dict
        Each dict contains: x_true, y_kspace, mask, coil_maps, B0_map,
        warp_field, metadata, true_spec, spec_ranges, idx.
    """
    import h5py

    samples: List[Dict[str, Any]] = []
    if not h5_path.exists():
        logger.warning("HDF5 file not found: %s", h5_path)
        return samples

    with h5py.File(h5_path, "r") as hf:
        group_names = sorted([k for k in hf.keys() if k.startswith("sample_")])
        if max_samples > 0:
            group_names = group_names[:max_samples]

        for gname in group_names:
            grp = hf[gname]
            sample: Dict[str, Any] = {
                "idx": int(gname.split("_")[1]),
                "x_true": grp["x_true"][:].astype(np.float32),
                "y_kspace": grp["y_kspace"][:].astype(np.complex64),
                "mask": grp["mask"][:],
                "coil_maps": grp["coil_maps"][:].astype(np.complex64),
                "B0_map": grp["B0_map"][:].astype(np.float32),
                "warp_field": grp["warp_field"][:].astype(np.float32),
            }
            # Parse JSON attributes
            for attr_name in ("metadata", "true_spec", "spec_ranges"):
                raw = grp.attrs.get(attr_name, "{}")
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                try:
                    sample[attr_name] = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    sample[attr_name] = {}
            samples.append(sample)

    logger.info("Loaded %d samples from %s", len(samples), h5_path.name)
    return samples


# ===========================================================================
# Extract mismatch info from metadata
# ===========================================================================
def _extract_mismatch(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Pull physics-mismatch parameters from sample metadata."""
    meta = sample.get("metadata", {})
    true_spec = sample.get("true_spec", {})
    mismatch: Dict[str, Any] = {}

    # B0 inhomogeneity
    b0 = sample.get("B0_map", None)
    if b0 is not None and b0.size > 0:
        mismatch["B0_inhomog_hz_max"] = round(float(np.max(np.abs(b0))), 2)
        mismatch["B0_inhomog_hz_std"] = round(float(np.std(b0)), 2)

    # Warp field magnitude
    wf = sample.get("warp_field", None)
    if wf is not None and wf.size > 0:
        mismatch["warp_field_max_px"] = round(float(np.max(np.abs(wf))), 2)

    # Copy any keys from metadata that look like mismatch parameters
    for key in ("acceleration", "noise_std", "coil_noise", "motion_level"):
        if key in meta:
            mismatch[key] = meta[key]
        if key in true_spec:
            mismatch[key] = true_spec[key]

    return mismatch


# ===========================================================================
# Aggregate statistics
# ===========================================================================
def _aggregate_solver_results(
    per_sample: List[Dict[str, Any]],
    solver_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute mean/std of PSNR, SSIM, and time across samples for each solver."""
    agg: Dict[str, Dict[str, float]] = {}
    for sname in solver_names:
        psnrs, ssims, times = [], [], []
        for s in per_sample:
            entry = s.get("solvers", {}).get(sname)
            if entry is None:
                continue
            if entry.get("psnr") is not None:
                psnrs.append(entry["psnr"])
            if entry.get("ssim") is not None:
                ssims.append(entry["ssim"])
            if entry.get("time_s") is not None:
                times.append(entry["time_s"])

        if not psnrs:
            continue

        agg[sname] = {
            "mean_psnr": round(float(np.mean(psnrs)), 4),
            "std_psnr": round(float(np.std(psnrs)), 4),
            "mean_ssim": round(float(np.mean(ssims)), 4) if ssims else 0.0,
            "std_ssim": round(float(np.std(ssims)), 4) if ssims else 0.0,
            "mean_time_s": round(float(np.mean(times)), 4) if times else 0.0,
            "n_samples": len(psnrs),
        }
    return agg


# ===========================================================================
# Cross-tier summary
# ===========================================================================
def _build_cross_tier_summary(
    tier_results: Dict[str, Dict[str, Any]],
    solver_names: List[str],
) -> Dict[str, Dict[str, Optional[float]]]:
    """Build cross-tier comparison: solver -> {tier_psnr, tier_ssim}."""
    summary: Dict[str, Dict[str, Optional[float]]] = {}
    for sname in solver_names:
        row: Dict[str, Optional[float]] = {}
        for tier_name, tier_data in tier_results.items():
            agg = tier_data.get("aggregate", {}).get(sname)
            if agg is not None:
                row[f"{tier_name}_psnr"] = agg.get("mean_psnr")
                row[f"{tier_name}_ssim"] = agg.get("mean_ssim")
            else:
                row[f"{tier_name}_psnr"] = None
                row[f"{tier_name}_ssim"] = None
        if any(v is not None for v in row.values()):
            summary[sname] = row
    return summary


# ===========================================================================
# Pretty-print helpers
# ===========================================================================
def _print_tier_summary(tier_name: str, tier_data: Dict[str, Any]) -> None:
    """Print a formatted table for one tier."""
    agg = tier_data.get("aggregate", {})
    n = tier_data.get("n_samples", 0)
    logger.info("")
    logger.info("=" * 72)
    logger.info("  TIER: %-8s  (%d samples)", tier_name.upper(), n)
    logger.info("=" * 72)
    logger.info(
        "  %-18s  %8s  %8s  %8s  %8s  %8s",
        "Solver", "PSNR", "std", "SSIM", "std", "Time(s)",
    )
    logger.info("  " + "-" * 68)
    for sname in SOLVER_NAMES:
        entry = agg.get(sname)
        if entry is None:
            continue
        logger.info(
            "  %-18s  %8.2f  %8.2f  %8.4f  %8.4f  %8.3f",
            sname,
            entry["mean_psnr"],
            entry["std_psnr"],
            entry["mean_ssim"],
            entry["std_ssim"],
            entry["mean_time_s"],
        )
    logger.info("  " + "-" * 68)


def _print_cross_tier(summary: Dict[str, Dict[str, Optional[float]]]) -> None:
    """Print cross-tier PSNR comparison."""
    logger.info("")
    logger.info("=" * 72)
    logger.info("  CROSS-TIER PSNR SUMMARY")
    logger.info("=" * 72)
    header_parts = ["  %-18s" % "Solver"]
    for t in ("public", "dev", "hidden"):
        header_parts.append("%10s" % t.upper())
    logger.info("".join(header_parts))
    logger.info("  " + "-" * 52)
    for sname in SOLVER_NAMES:
        row = summary.get(sname)
        if row is None:
            continue
        parts = ["  %-18s" % sname]
        for t in ("public", "dev", "hidden"):
            val = row.get(f"{t}_psnr")
            parts.append("%10s" % (f"{val:.2f}" if val is not None else "N/A"))
        logger.info("".join(parts))
    logger.info("  " + "-" * 52)


# ===========================================================================
# Auto-detect device
# ===========================================================================
def _auto_device() -> str:
    """Return 'cuda' if available, else 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


# ===========================================================================
# Main entry point
# ===========================================================================
def run_mri_multiphantom(
    tiers_to_run: List[str],
    solvers_to_run: List[str],
    max_samples: int = 0,
    device: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the multi-algorithm MRI benchmark.

    Parameters
    ----------
    tiers_to_run : list of str
        Tier names to evaluate (subset of 'public', 'dev', 'hidden').
    solvers_to_run : list of str
        Internal solver names to evaluate.
    max_samples : int
        Cap on samples per tier (0 = all).
    device : str or None
        Compute device ('cpu', 'cuda', or None for auto-detect).
    output_path : Path or None
        Where to write the JSON results (None = default location).

    Returns
    -------
    dict
        Full results structure.
    """
    if device is None:
        device = _auto_device()
    logger.info("Device: %s", device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = RESULTS_DIR / "mri_multiphantom_results.json"

    timestamp = datetime.now(timezone.utc).isoformat()

    logger.info("")
    logger.info("=" * 72)
    logger.info("  MRI MULTI-ALGORITHM BENCHMARK")
    logger.info("=" * 72)
    logger.info("  Tiers   : %s", ", ".join(tiers_to_run))
    logger.info("  Solvers : %s", ", ".join(solvers_to_run))
    logger.info("  Max smpl: %s", max_samples if max_samples > 0 else "all")
    logger.info("  Device  : %s", device)
    logger.info("  Output  : %s", output_path)
    logger.info("=" * 72)

    tier_results: Dict[str, Dict[str, Any]] = {}
    first_shape = None
    first_n_coils = None

    for tier_name in tiers_to_run:
        h5_path = TIERS.get(tier_name)
        if h5_path is None:
            logger.warning("Unknown tier '%s', skipping.", tier_name)
            continue

        logger.info("")
        logger.info("#" * 72)
        logger.info("# TIER: %s", tier_name.upper())
        logger.info("#   File: %s", h5_path)
        logger.info("#" * 72)

        samples = load_samples(h5_path, max_samples=max_samples)
        if not samples:
            logger.warning("No samples loaded for tier '%s'. Skipping.", tier_name)
            continue

        per_sample_results: List[Dict[str, Any]] = []

        for si, sample in enumerate(samples):
            idx = sample["idx"]
            x_true = sample["x_true"]
            y_kspace = sample["y_kspace"]
            mask_1d = sample["mask"]
            coil_maps = sample["coil_maps"]

            shape = tuple(x_true.shape)
            n_coils = y_kspace.shape[0]
            if first_shape is None:
                first_shape = shape
                first_n_coils = n_coils

            scene_name = sample.get("metadata", {}).get("scene", f"sample_{idx:02d}")
            mismatch = _extract_mismatch(sample)

            logger.info("")
            logger.info(
                "  [%s] Sample %d/%d  idx=%d  scene=%s  shape=%s  coils=%d  mask_sum=%d/%d",
                tier_name,
                si + 1,
                len(samples),
                idx,
                scene_name,
                shape,
                n_coils,
                int(mask_1d.astype(bool).sum()),
                mask_1d.shape[0],
            )

            sample_result: Dict[str, Any] = {
                "idx": idx,
                "scene": str(scene_name),
                "mismatch": mismatch,
                "solvers": {},
            }

            for sname in solvers_to_run:
                solve_fn = SOLVER_DISPATCH.get(sname)
                if solve_fn is None:
                    logger.warning("  Unknown solver '%s', skipping.", sname)
                    continue

                logger.info("    Running %-18s ...", sname)
                t0 = time.time()
                try:
                    x_hat, info = solve_fn(x_true, y_kspace, mask_1d, coil_maps, device)
                except Exception:
                    tb = traceback.format_exc()
                    logger.error("    FAILED %s: %s", sname, tb.splitlines()[-1])
                    logger.info("    Falling back to zero-filled RSS.")
                    try:
                        x_hat = zero_filled_reconstruction(y_kspace, device=device)
                        x_hat = _normalise_to_01(x_hat)
                    except Exception:
                        x_hat = np.zeros_like(x_true)
                    info = {"solver": sname, "error": tb.splitlines()[-1], "fallback": "zerofilled_rss"}
                elapsed = time.time() - t0

                # Ensure x_hat matches x_true shape
                if x_hat.shape != x_true.shape:
                    logger.warning(
                        "    Shape mismatch: x_hat=%s vs x_true=%s. Attempting resize.",
                        x_hat.shape,
                        x_true.shape,
                    )
                    try:
                        from scipy.ndimage import zoom
                        factors = tuple(t / h for t, h in zip(x_true.shape, x_hat.shape))
                        x_hat = zoom(x_hat, factors, order=1).astype(np.float32)
                    except Exception:
                        x_hat = np.zeros_like(x_true)

                # Compute metrics
                psnr_val = compute_psnr(x_true, x_hat, max_val=1.0)
                ssim_val = compute_ssim(x_true, x_hat, data_range=1.0)

                sample_result["solvers"][sname] = {
                    "psnr": round(float(psnr_val), 4),
                    "ssim": round(float(ssim_val), 4),
                    "time_s": round(elapsed, 4),
                }
                if "error" in info:
                    sample_result["solvers"][sname]["error"] = info["error"]
                if "fallback" in info:
                    sample_result["solvers"][sname]["fallback"] = info["fallback"]
                if "note" in info:
                    sample_result["solvers"][sname]["note"] = info["note"]

                logger.info(
                    "    %-18s  PSNR=%6.2f dB  SSIM=%.4f  time=%.3f s",
                    sname,
                    psnr_val,
                    ssim_val,
                    elapsed,
                )

            per_sample_results.append(sample_result)

        # Aggregate for this tier
        aggregate = _aggregate_solver_results(per_sample_results, solvers_to_run)

        tier_results[tier_name] = {
            "n_samples": len(per_sample_results),
            "per_sample": per_sample_results,
            "aggregate": aggregate,
        }

        _print_tier_summary(tier_name, tier_results[tier_name])

    # Cross-tier summary
    cross_tier = _build_cross_tier_summary(tier_results, solvers_to_run)
    if len(tier_results) > 1:
        _print_cross_tier(cross_tier)

    # Determine acceleration factor from mask
    accel = None
    for tier_name in tiers_to_run:
        h5_path = TIERS.get(tier_name)
        if h5_path is not None and h5_path.exists():
            try:
                import h5py
                with h5py.File(h5_path, "r") as hf:
                    first_grp = sorted(k for k in hf.keys() if k.startswith("sample_"))[0]
                    m = hf[first_grp]["mask"][:]
                    accel = round(float(m.shape[0]) / float(m.astype(bool).sum()), 1)
                break
            except Exception:
                pass

    # Assemble final results
    results: Dict[str, Any] = {
        "modality": "mri",
        "timestamp": timestamp,
        "shape": list(first_shape) if first_shape else [320, 320],
        "n_coils": first_n_coils if first_n_coils else 15,
        "acceleration": accel if accel else 4,
        "device": device,
        "solvers_evaluated": solvers_to_run,
        "tiers": tier_results,
        "cross_tier_summary": cross_tier,
    }

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("")
    logger.info("Results saved to %s", output_path)

    return results


# ===========================================================================
# CLI
# ===========================================================================
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MRI Multi-Algorithm Benchmark (multi-phantom, multi-tier)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tier",
        choices=["public", "dev", "hidden", "all"],
        default="all",
        help="Which data tier(s) to evaluate (default: all).",
    )
    parser.add_argument(
        "--solver",
        choices=list(SOLVER_CLI_MAP.keys()) + ["all"],
        default="all",
        help="Which solver(s) to run (default: all).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max samples per tier; 0 means all (default: 0).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Compute device; auto-detected if omitted.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: auto-generated in results dir).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    # Resolve tiers
    if args.tier == "all":
        tiers_to_run = ["public", "dev", "hidden"]
    else:
        tiers_to_run = [args.tier]

    # Resolve solvers
    if args.solver == "all":
        solvers_to_run = list(SOLVER_NAMES)
    else:
        internal_name = SOLVER_CLI_MAP.get(args.solver)
        if internal_name is None:
            logger.error("Unknown solver shorthand: %s", args.solver)
            sys.exit(1)
        solvers_to_run = [internal_name]

    # Resolve output path
    output_path = Path(args.output) if args.output else None

    run_mri_multiphantom(
        tiers_to_run=tiers_to_run,
        solvers_to_run=solvers_to_run,
        max_samples=args.max_samples,
        device=args.device,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
