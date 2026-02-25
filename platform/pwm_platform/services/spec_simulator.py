"""Spec Simulator — runs the full forward → noise → reconstruct → metrics pipeline.

Given a spec JSON (from the chat builder) and a variant key, this module:
  1. Maps modality (cassi / spc / cacti)
  2. Loads canonical benchmark data
  3. Builds the physics operator
  4. Runs the forward model
  5. Applies noise
  6. Reconstructs the signal
  7. Computes PSNR / SSIM metrics
  8. Runs bottleneck analysis
  9. Saves result images
 10. Returns a SimulationResult dataclass
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
SIM_IMG_DIR = STATIC_DIR / "simulations"


# ── Result dataclass ──────────────────────────────────────────────────────


@dataclass
class SimulationResult:
    ground_truth_path: str
    measurement_path: str
    reconstructed_path: str
    psnr: float
    ssim: float
    solver_name: str
    solver_iters: int
    bottleneck: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    dataset_name: str
    sim_id: str


# ── Image saving (reuses pattern from demo_images.py) ─────────────────────


def _ensure_dir(sim_id: str) -> Path:
    d = SIM_IMG_DIR / sim_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_image(
    arr: np.ndarray,
    path: Path,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: str = "",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=100)
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9, color="#333", pad=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)


# ── SSIM (lightweight implementation) ─────────────────────────────────────


def _ssim_2d(x: np.ndarray, y: np.ndarray, data_range: float = 1.0) -> float:
    """Compute SSIM between two 2D images (mean SSIM for 3D)."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    x = x.astype(np.float64)
    y = y.astype(np.float64)

    # For 3D data, average SSIM over last axis
    if x.ndim == 3:
        vals = [_ssim_2d(x[:, :, i], y[:, :, i], data_range) for i in range(x.shape[2])]
        return float(np.mean(vals))

    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x2 = ((x - mu_x) ** 2).mean()
    sigma_y2 = ((y - mu_y) ** 2).mean()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return float(num / den)


# ── Noise application (simple, no dependency on SensorState) ──────────────


def _apply_noise(y_clean: np.ndarray, spec: dict, rng: np.random.Generator) -> np.ndarray:
    """Apply noise based on the spec's noise_model description string."""
    noise_desc = (spec.get("noise_model") or "").lower()
    y = y_clean.copy()

    if "poisson" in noise_desc and "gaussian" in noise_desc:
        # Poisson-Gaussian mixed
        gain = 100.0
        sigma = 0.02
        y_pos = np.clip(y, 0, None)
        y = rng.poisson(np.clip(y_pos * gain, 0, 1e8)).astype(np.float64) / gain
        y += rng.normal(0, sigma, y.shape)
    elif "poisson" in noise_desc:
        gain = _parse_gain(noise_desc, default=100.0)
        y_pos = np.clip(y, 0, None)
        y = rng.poisson(np.clip(y_pos * gain, 0, 1e8)).astype(np.float64) / gain
    elif "gaussian" in noise_desc:
        sigma = _parse_sigma(noise_desc, default=0.02)
        y += rng.normal(0, sigma, y.shape)
    else:
        # Default mild Gaussian
        y += rng.normal(0, 0.02, y.shape)

    return y.astype(np.float32)


def _parse_gain(desc: str, default: float = 100.0) -> float:
    """Try to extract gain=XX from noise description."""
    import re
    m = re.search(r"gain\s*[=:]\s*([0-9.]+)", desc)
    return float(m.group(1)) if m else default


def _parse_sigma(desc: str, default: float = 0.02) -> float:
    """Try to extract sigma=XX or σ=XX from noise description."""
    import re
    m = re.search(r"(?:sigma|σ)\s*[=:]\s*([0-9.]+)", desc)
    return float(m.group(1)) if m else default


# ── Modality classification ───────────────────────────────────────────────


def _classify_modality(variant_key: str, spec: Optional[dict] = None) -> str:
    """Map variant_key + spec content to one of: cassi, spc, cacti.

    The variant_key alone may not distinguish modalities (e.g. all examples
    may share the same variant_key). We also inspect the spec's notation,
    measurement_matrix, and forward_model primitives.
    """
    vk = variant_key.lower()

    # First, try to detect from spec content (most reliable for examples)
    if spec:
        notation = (spec.get("spec_notation") or "").lower()
        meas_matrix = (spec.get("measurement_matrix") or "").lower()
        labels = " ".join(
            (n.get("label") or "").lower() for n in (spec.get("forward_model") or [])
        )
        all_text = f"{notation} {meas_matrix} {labels}"

        if "spc" in all_text or "single-pixel" in all_text or "block" in all_text or "sensing matrix" in meas_matrix:
            return "spc"
        if "cacti" in all_text or "temporal" in all_text or "video" in all_text or "time-varying" in meas_matrix:
            return "cacti"
        if "cassi" in all_text or "dispersion" in all_text or "spectral" in all_text:
            return "cassi"

    # Fall back to variant_key
    if "spc" in vk or "single_pixel" in vk:
        return "spc"
    if "cacti" in vk:
        return "cacti"
    if "cassi" in vk:
        return "cassi"

    return "cassi"


# ── Per-modality simulation pipelines ─────────────────────────────────────


def _run_cassi(spec: dict, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, str, int]:
    """Run CASSI simulation pipeline. Returns (x_true, y_noisy, x_hat, dataset_name, solver_name, iters)."""
    from pwm_core.data.loaders.kaist import KAISTDataset
    from pwm_core.physics.spectral.cassi_operator import SDCASSIOperator
    from pwm_core.recon.cs_solvers import run_tval3

    # Load a random scene from the 10-scene KAIST benchmark dataset
    dataset = KAISTDataset(resolution=256, num_bands=28)
    scenes = list(dataset)
    idx = int(rng.integers(0, len(scenes)))
    name, cube = scenes[idx]

    # Build operator
    H, W, L = cube.shape
    mask = (rng.random((H, W)) > 0.5).astype(np.float32)
    operator = SDCASSIOperator(
        operator_id="sd_cassi",
        theta={"L": L, "dispersion_step": 2.0},
        mask=mask,
    )
    # Set x_shape so run_tval3 can determine reconstruction shape
    operator.x_shape = (H, W, L)

    # Forward model
    y_clean = operator.forward(cube)

    # Apply noise
    y_noisy = _apply_noise(y_clean, spec, rng)

    # Reconstruct with TVAL3
    iters = 100
    cfg = {"mu": 256, "beta": 32, "iters": iters}
    x_hat, info = run_tval3(y_noisy, operator, cfg)

    # Ensure x_hat has correct shape for metrics
    if x_hat.ndim == 1:
        x_hat = x_hat.reshape(cube.shape)
    elif x_hat.shape != cube.shape:
        # Best effort reshape
        try:
            x_hat = x_hat.reshape(cube.shape)
        except ValueError:
            x_hat = operator.adjoint(y_noisy)

    return cube, y_noisy, x_hat, name, "tval3", iters


def _run_spc(spec: dict, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, str, int]:
    """Run SPC simulation pipeline."""
    from pwm_core.data.loaders.set11 import Set11Dataset
    from pwm_core.physics.compressive.spc_operator import SPCOperator
    from pwm_core.recon.cs_solvers import run_admm_tv

    # Load a random image from the 11-image Set11 benchmark dataset
    dataset = Set11Dataset(resolution=64)
    images = list(dataset)
    idx = int(rng.integers(0, len(images)))
    name, img = images[idx]

    # Build operator
    operator = SPCOperator(x_shape=(64, 64), sampling_rate=0.15)

    # Forward model
    y_clean = operator.forward(img)

    # Apply noise
    y_noisy = _apply_noise(y_clean, spec, rng)

    # Reconstruct with ADMM-TV
    iters = 500
    x_hat, info = run_admm_tv(y_noisy, operator.A, (64, 64))

    # Reshape if needed
    if x_hat.ndim == 1:
        x_hat = x_hat.reshape(64, 64)

    return img, y_noisy, x_hat, name, "admm_tv", iters


def _run_cacti(spec: dict, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, str, int]:
    """Run CACTI simulation pipeline with synthetic fallback."""
    from pwm_core.physics.compressive.cacti_operator import CACTIOperator
    from pwm_core.recon.cacti_solvers import gap_tv_cacti

    # Try real benchmark data first, fall back to 6-video synthetic dataset
    x_true, mask_3d, dataset_name = _load_cacti_data(rng)

    H, W, T = x_true.shape

    # Build operator (use the loaded mask)
    operator = CACTIOperator(
        x_shape=(H, W, T),
        mask=mask_3d[:, :, 0],  # Base mask for the operator
        shift_type="vertical",
    )
    # Override operator masks with the real masks if available
    operator.masks = mask_3d

    # Forward model
    y_clean = operator.forward(x_true)

    # Apply noise
    y_noisy = _apply_noise(y_clean, spec, rng)

    # Reconstruct with GAP-TV
    iters = 100
    x_hat = gap_tv_cacti(y_noisy, mask_3d, iterations=iters)

    return x_true, y_noisy, x_hat, dataset_name, "gap_tv", iters


def _load_cacti_data(rng: np.random.Generator = None) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load CACTI benchmark data, falling back to synthetic dataset if .mat files unavailable."""
    try:
        from pwm_core.data.loaders.cacti_bench import CACTIBenchmark
        bench = CACTIBenchmark()
        groups = list(bench)
        if rng is not None:
            idx = int(rng.integers(0, len(groups)))
        else:
            idx = 0
        name, group_gt, mask, meas = groups[idx]
        return group_gt, mask, name
    except (FileNotFoundError, Exception) as exc:
        logger.info("CACTI benchmark data unavailable (%s), using SyntheticCACTIDataset", exc)
        return _synthetic_cacti(rng)


def _synthetic_cacti(rng: np.random.Generator = None) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load from the 6-video SyntheticCACTIDataset benchmark."""
    from pwm_core.data.loaders.cacti_bench import SyntheticCACTIDataset

    dataset = SyntheticCACTIDataset(resolution=256, num_frames=8)
    videos = list(dataset)
    if rng is not None:
        idx = int(rng.integers(0, len(videos)))
    else:
        idx = 0
    name, video, mask = videos[idx]
    return video, mask, name


# ── Bottleneck estimation from spec + metrics ─────────────────────────────


def _estimate_bottleneck(
    spec: dict,
    psnr_val: float,
    modality: str,
    solver_name: str,
) -> Dict[str, Any]:
    """Estimate bottleneck severities from the spec and simulation results."""
    from pwm_core.analysis.bottleneck import rank_bottlenecks

    # Estimate photon severity from noise model
    noise_desc = (spec.get("noise_model") or "").lower()
    photon_sev = 0.2  # default mild
    if "poisson" in noise_desc:
        gain = _parse_gain(noise_desc, 100.0)
        photon_sev = min(1.0, max(0.1, 1.0 - gain / 200.0))
    if "high" in noise_desc or "severe" in noise_desc:
        photon_sev = 0.7

    # Estimate mismatch severity from mismatch params
    mismatch_params = spec.get("mismatch_params", [])
    mismatch_sev = 0.0
    if mismatch_params:
        deltas = []
        for p in mismatch_params:
            nom = float(p.get("nominal", 0))
            pert = float(p.get("perturbed", 0))
            if nom != 0:
                deltas.append(abs(pert - nom) / abs(nom))
            elif pert != 0:
                deltas.append(min(abs(pert), 1.0))
        if deltas:
            mismatch_sev = min(1.0, np.mean(deltas))

    # Estimate recoverability from PSNR
    recov_sev = max(0.0, min(1.0, (30.0 - psnr_val) / 20.0))

    # Solver severity: low for known good solvers, moderate otherwise
    solver_sev = 0.2 if solver_name in ("tval3", "admm_tv", "gap_tv") else 0.4

    # Get compression ratio from modality
    cr = {"cassi": 28.0, "spc": 6.7, "cacti": 8.0}.get(modality, 8.0)

    mismatch_family = None
    if mismatch_params:
        mismatch_family = mismatch_params[0].get("name", "unknown")

    return rank_bottlenecks(
        photon_severity=photon_sev,
        recoverability_severity=recov_sev,
        mismatch_severity=mismatch_sev,
        solver_fit_severity=solver_sev,
        snr_db=psnr_val,
        compression_ratio=cr,
        mismatch_family=mismatch_family,
        solver_family=solver_name,
    )


# ── Image saving helpers ──────────────────────────────────────────────────


def _select_display_slice(arr: np.ndarray) -> np.ndarray:
    """Pick a 2D slice for display from a possibly 3D array."""
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # For hyperspectral: show middle band; for video: show first frame
        mid = arr.shape[2] // 2
        return arr[:, :, mid]
    return arr.reshape(arr.shape[0], -1)


def _save_simulation_images(
    sim_id: str,
    x_true: np.ndarray,
    y_noisy: np.ndarray,
    x_hat: np.ndarray,
    modality: str,
) -> Tuple[str, str, str]:
    """Save ground truth, measurement, and reconstruction images. Returns URL paths."""
    d = _ensure_dir(sim_id)

    gt_slice = _select_display_slice(x_true)
    recon_slice = _select_display_slice(x_hat)

    # Measurement may be 1D (SPC) or 2D
    if y_noisy.ndim == 1:
        # Reshape 1D SPC measurement into a square-ish image for display
        n = len(y_noisy)
        side = int(np.ceil(np.sqrt(n)))
        meas_display = np.zeros(side * side, dtype=np.float32)
        meas_display[:n] = y_noisy
        meas_display = meas_display.reshape(side, side)
    else:
        meas_display = y_noisy

    cmap = "viridis" if modality == "cassi" else "gray"

    _save_image(gt_slice, d / "ground_truth.png", cmap=cmap, title="Ground Truth")
    _save_image(meas_display, d / "measurement.png", cmap="inferno", title="Measurement")
    _save_image(recon_slice, d / "reconstructed.png", cmap=cmap, title="Reconstructed")

    base = f"/static/simulations/{sim_id}"
    return f"{base}/ground_truth.png", f"{base}/measurement.png", f"{base}/reconstructed.png"


# ── Main entry point ──────────────────────────────────────────────────────


async def run_spec_simulation(spec: dict, variant_key: str) -> SimulationResult:
    """Run a full simulation pipeline for the given spec and variant.

    This is an async wrapper around the CPU-bound simulation work,
    executed in a thread pool to avoid blocking the event loop.
    """
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run_simulation_sync, spec, variant_key)


def _run_simulation_sync(spec: dict, variant_key: str) -> SimulationResult:
    """Synchronous simulation pipeline."""
    sim_id = uuid.uuid4().hex[:12]
    rng = np.random.default_rng(42)

    modality = _classify_modality(variant_key, spec)
    logger.info("Running %s simulation (sim_id=%s)", modality, sim_id)

    # Run modality-specific pipeline
    runners = {
        "cassi": _run_cassi,
        "spc": _run_spc,
        "cacti": _run_cacti,
    }
    runner = runners.get(modality, _run_cassi)
    x_true, y_noisy, x_hat, dataset_name, solver_name, solver_iters = runner(spec, rng)

    # Compute metrics
    from pwm_core.analysis.metrics import psnr as compute_psnr

    # Ensure matching shapes for metrics
    if x_hat.shape != x_true.shape:
        try:
            x_hat = x_hat.reshape(x_true.shape)
        except ValueError:
            logger.warning("Shape mismatch: x_true=%s, x_hat=%s", x_true.shape, x_hat.shape)

    x_true_clipped = np.clip(x_true.astype(np.float64), 0, 1)
    x_hat_clipped = np.clip(x_hat.astype(np.float64), 0, 1)

    psnr_val = compute_psnr(x_true_clipped, x_hat_clipped, data_range=1.0)
    ssim_val = _ssim_2d(x_true_clipped, x_hat_clipped, data_range=1.0)

    # Save images
    gt_path, meas_path, recon_path = _save_simulation_images(
        sim_id, x_true, y_noisy, x_hat, modality
    )

    # Bottleneck analysis
    bottleneck = _estimate_bottleneck(spec, psnr_val, modality, solver_name)

    return SimulationResult(
        ground_truth_path=gt_path,
        measurement_path=meas_path,
        reconstructed_path=recon_path,
        psnr=round(psnr_val, 2),
        ssim=round(ssim_val, 4),
        solver_name=solver_name,
        solver_iters=solver_iters,
        bottleneck=bottleneck,
        recommendations=bottleneck.get("ranked", []),
        dataset_name=dataset_name,
        sim_id=sim_id,
    )
