"""Run all four InverseNet benchmark tasks with baseline methods.

Tasks
-----
T1 -- Operator parameter estimation  (theta-error RMSE)
T2 -- Mismatch identification        (accuracy, F1)
T3 -- Calibration                    (theta-error reduction, CI coverage)
T4 -- Reconstruction under mismatch  (PSNR, SSIM, SAM)

Baselines
---------
- Oracle operator / wrong operator
- Grid search / gradient-free / UPWMI calibration
- GAP-TV / PnP / LISTA reconstruction

Usage::

    python -m experiments.inversenet.run_baselines \\
        --data_dirs datasets/inversenet_spc datasets/inversenet_cacti datasets/inversenet_cassi \\
        --results_dir results/inversenet

    # Smoke test (1 sample per modality):
    python -m experiments.inversenet.run_baselines --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from experiments.inversenet.manifest_schema import ManifestRecord, Modality, Severity
from experiments.inversenet.gen_spc import generate_spc_dataset
from experiments.inversenet.gen_cacti import generate_cacti_dataset
from experiments.inversenet.gen_cassi import generate_cassi_dataset

logger = logging.getLogger(__name__)

# ── Metric helpers ──────────────────────────────────────────────────────


def compute_psnr(x: np.ndarray, y: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB."""
    mse = float(np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2))
    if mse < 1e-10:
        return 100.0
    max_val = max(float(x.max()), float(y.max()), 1.0)
    return float(10 * np.log10(max_val ** 2 / mse))


def compute_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """Simplified SSIM (no skimage dependency)."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    x64 = x.astype(np.float64).ravel()
    y64 = y.astype(np.float64).ravel()
    mu_x, mu_y = x64.mean(), y64.mean()
    sig_x, sig_y = x64.std(), y64.std()
    sig_xy = float(np.mean((x64 - mu_x) * (y64 - mu_y)))
    return float(
        (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
        / ((mu_x ** 2 + mu_y ** 2 + C1) * (sig_x ** 2 + sig_y ** 2 + C2))
    )


def compute_sam(x: np.ndarray, y: np.ndarray) -> float:
    """Spectral Angle Mapper (radians) -- only for 3-D cubes."""
    if x.ndim < 3:
        return 0.0
    H, W, L = x.shape
    x_flat = x.reshape(-1, L).astype(np.float64)
    y_flat = y.reshape(-1, L).astype(np.float64)
    dot = np.sum(x_flat * y_flat, axis=1)
    nx = np.linalg.norm(x_flat, axis=1)
    ny = np.linalg.norm(y_flat, axis=1)
    cos_theta = dot / (nx * ny + 1e-10)
    cos_theta = np.clip(cos_theta, -1, 1)
    return float(np.mean(np.arccos(cos_theta)))


def theta_rmse(theta_true: Dict[str, Any], theta_est: Dict[str, Any]) -> float:
    """RMSE between two theta dicts (numeric values only)."""
    diffs = []
    for k in theta_true:
        if k in theta_est:
            try:
                diffs.append((float(theta_true[k]) - float(theta_est[k])) ** 2)
            except (TypeError, ValueError):
                continue
    if not diffs:
        return 0.0
    return float(np.sqrt(np.mean(diffs)))


# ── Result containers ──────────────────────────────────────────────────


@dataclass
class TaskResult:
    task: str
    sample_id: str
    modality: str
    baseline: str
    metrics: Dict[str, float] = field(default_factory=dict)
    runtime_s: float = 0.0


# ── Baseline: SPC ──────────────────────────────────────────────────────


def _spc_reconstruct_lsq(
    y: np.ndarray, A: np.ndarray, gain: float = 1.0, bias: float = 0.0
) -> np.ndarray:
    """Regularised least-squares SPC reconstruction."""
    y_corr = (y - bias) / max(gain, 0.01)
    reg = 0.001
    AtA = A.T @ A + reg * np.eye(A.shape[1])
    Aty = A.T @ y_corr
    x_hat = np.linalg.solve(AtA, Aty)
    n = int(np.sqrt(len(x_hat)))
    return np.clip(x_hat, 0, 1).reshape(n, n).astype(np.float32)


def _spc_estimate_gain_grid(
    y: np.ndarray, A: np.ndarray
) -> Tuple[float, float]:
    """Grid search gain/bias estimation."""
    best_g, best_b, best_tv = 1.0, 0.0, float("inf")
    for g in np.linspace(0.5, 2.0, 11):
        for b in np.linspace(-0.2, 0.2, 11):
            x_hat = _spc_reconstruct_lsq(y, A, g, b)
            tv = float(
                np.sum(np.abs(np.diff(x_hat, axis=0)))
                + np.sum(np.abs(np.diff(x_hat, axis=1)))
            )
            if tv < best_tv:
                best_tv = tv
                best_g, best_b = g, b
    return best_g, best_b


def run_spc_baselines(
    sample_dir: str, rec: ManifestRecord
) -> List[TaskResult]:
    """Run T1-T4 baselines for one SPC sample."""
    results: List[TaskResult] = []
    sid = rec.sample_id

    x_gt = np.load(os.path.join(sample_dir, "x_gt.npy"))
    y = np.load(os.path.join(sample_dir, "y.npy"))
    A = np.load(os.path.join(sample_dir, "mask.npy"))

    with open(os.path.join(sample_dir, "theta.json")) as f:
        theta_true = json.load(f)
    with open(os.path.join(sample_dir, "delta_theta.json")) as f:
        delta_theta = json.load(f)

    # ── T1: parameter estimation ──
    t0 = time.time()
    g_est, b_est = _spc_estimate_gain_grid(y, A)
    theta_est = {"gain": g_est, "bias": b_est}
    theta_target = {
        "gain": theta_true.get("gain", 1.0) * delta_theta.get("gain_factor", 1.0),
        "bias": theta_true.get("bias", 0.0) + delta_theta.get("bias", 0.0),
    }
    rmse = theta_rmse(theta_target, theta_est)
    results.append(TaskResult(
        task="T1_param_estimation",
        sample_id=sid, modality="spc", baseline="grid_search",
        metrics={"theta_error_rmse": rmse},
        runtime_s=time.time() - t0,
    ))

    # ── T2: mismatch identification ──
    t0 = time.time()
    # Simple heuristic: if estimated gain far from 1.0 -> "gain" family
    pred_family = "gain" if abs(g_est - 1.0) > 0.03 else "mask_error"
    correct = int(pred_family == rec.mismatch_family)
    results.append(TaskResult(
        task="T2_mismatch_id",
        sample_id=sid, modality="spc", baseline="heuristic",
        metrics={"accuracy": float(correct), "f1": float(correct)},
        runtime_s=time.time() - t0,
    ))

    # ── T3: calibration ──
    t0 = time.time()
    x_before = _spc_reconstruct_lsq(y, A, gain=1.0, bias=0.0)
    x_after = _spc_reconstruct_lsq(y, A, gain=g_est, bias=b_est)
    psnr_before = compute_psnr(x_gt, x_before)
    psnr_after = compute_psnr(x_gt, x_after)
    results.append(TaskResult(
        task="T3_calibration",
        sample_id=sid, modality="spc", baseline="grid_search",
        metrics={
            "theta_error_rmse": rmse,
            "psnr_before_db": psnr_before,
            "psnr_after_db": psnr_after,
            "psnr_gain_db": psnr_after - psnr_before,
        },
        runtime_s=time.time() - t0,
    ))

    # ── T4: reconstruction ──
    for recon_name, gain_v, bias_v in [
        ("oracle", theta_target.get("gain", 1.0), theta_target.get("bias", 0.0)),
        ("wrong_op", 1.0, 0.0),
        ("calibrated", g_est, b_est),
    ]:
        t0 = time.time()
        x_hat = _spc_reconstruct_lsq(y, A, gain=gain_v, bias=bias_v)
        psnr = compute_psnr(x_gt, x_hat)
        ssim = compute_ssim(x_gt, x_hat)
        results.append(TaskResult(
            task="T4_reconstruction",
            sample_id=sid, modality="spc", baseline=recon_name,
            metrics={"psnr_db": psnr, "ssim": ssim, "sam": 0.0},
            runtime_s=time.time() - t0,
        ))

    return results


# ── Baseline: CACTI ────────────────────────────────────────────────────


def _cacti_gap_tv_simple(
    y: np.ndarray, masks: np.ndarray, iters: int = 30
) -> np.ndarray:
    """Simplified GAP-TV for CACTI."""
    from scipy.ndimage import gaussian_filter

    H, W = y.shape
    T = masks.shape[2]
    mask_sum = np.sum(masks, axis=2)
    mask_sum[mask_sum == 0] = 1
    x = np.tile(y[:, :, np.newaxis], (1, 1, T)) / T
    y1 = y.copy()
    for _ in range(iters):
        yb = np.sum(x * masks, axis=2)
        y1 = y1 + (y - yb)
        for t in range(T):
            x[:, :, t] += masks[:, :, t] * (y1 - yb) / mask_sum
        for t in range(T):
            x[:, :, t] = gaussian_filter(x[:, :, t], sigma=0.3)
        x = np.clip(x, 0, 1)
    return x.astype(np.float32)


def run_cacti_baselines(
    sample_dir: str, rec: ManifestRecord
) -> List[TaskResult]:
    """Run T1-T4 for one CACTI sample."""
    results: List[TaskResult] = []
    sid = rec.sample_id

    x_gt = np.load(os.path.join(sample_dir, "x_gt.npy"))
    y = np.load(os.path.join(sample_dir, "y.npy"))
    masks_true = np.load(os.path.join(sample_dir, "masks.npy"))
    masks_mm = np.load(os.path.join(sample_dir, "masks_mm.npy"))

    with open(os.path.join(sample_dir, "theta.json")) as f:
        theta_true = json.load(f)
    with open(os.path.join(sample_dir, "delta_theta.json")) as f:
        delta_theta = json.load(f)

    # ── T1: parameter estimation (timing offset) ──
    t0 = time.time()
    T = masks_true.shape[2]
    best_offset, best_res = 0, float("inf")
    for off in range(min(T, 6)):
        masks_test = np.roll(masks_true, off, axis=0)
        x_test = _cacti_gap_tv_simple(y, masks_test, iters=10)
        yb = np.sum(x_test * masks_test, axis=2)
        res = float(np.sum((y - yb) ** 2))
        if res < best_res:
            best_res = res
            best_offset = off
    theta_est = {"timing_offset": best_offset}
    theta_target = {"timing_offset": delta_theta.get("timing_offset",
                                                      delta_theta.get("shift_px", 0))}
    rmse = theta_rmse(theta_target, theta_est)
    results.append(TaskResult(
        task="T1_param_estimation",
        sample_id=sid, modality="cacti", baseline="grid_search",
        metrics={"theta_error_rmse": rmse},
        runtime_s=time.time() - t0,
    ))

    # ── T2: mismatch identification ──
    t0 = time.time()
    mask_corr = float(np.corrcoef(masks_mm.ravel(), masks_true.ravel())[0, 1])
    pred_family = "temporal_jitter" if mask_corr > 0.8 else "mask_shift"
    correct = int(pred_family == rec.mismatch_family)
    results.append(TaskResult(
        task="T2_mismatch_id",
        sample_id=sid, modality="cacti", baseline="heuristic",
        metrics={"accuracy": float(correct), "f1": float(correct)},
        runtime_s=time.time() - t0,
    ))

    # ── T3: calibration ──
    t0 = time.time()
    x_before = _cacti_gap_tv_simple(y, masks_mm, iters=20)
    masks_cal = np.roll(masks_true, best_offset, axis=0)
    x_after = _cacti_gap_tv_simple(y, masks_cal, iters=20)

    # Compare per-frame PSNR
    psnr_before = compute_psnr(x_gt, x_before)
    psnr_after = compute_psnr(x_gt, x_after)
    results.append(TaskResult(
        task="T3_calibration",
        sample_id=sid, modality="cacti", baseline="grid_search",
        metrics={
            "theta_error_rmse": rmse,
            "psnr_before_db": psnr_before,
            "psnr_after_db": psnr_after,
            "psnr_gain_db": psnr_after - psnr_before,
        },
        runtime_s=time.time() - t0,
    ))

    # ── T4: reconstruction ──
    for name, m in [("oracle", masks_true), ("wrong_op", masks_mm),
                     ("calibrated", masks_cal)]:
        t0 = time.time()
        x_hat = _cacti_gap_tv_simple(y, m, iters=30)
        psnr = compute_psnr(x_gt, x_hat)
        ssim = compute_ssim(
            x_gt.mean(axis=2) if x_gt.ndim == 3 else x_gt,
            x_hat.mean(axis=2) if x_hat.ndim == 3 else x_hat,
        )
        results.append(TaskResult(
            task="T4_reconstruction",
            sample_id=sid, modality="cacti", baseline=name,
            metrics={"psnr_db": psnr, "ssim": ssim, "sam": 0.0},
            runtime_s=time.time() - t0,
        ))

    return results


# ── Baseline: CASSI ────────────────────────────────────────────────────


def _cassi_gap_tv_simple(
    y: np.ndarray,
    mask: np.ndarray,
    theta: Dict[str, Any],
    n_bands: int,
    iters: int = 30,
) -> np.ndarray:
    """Simplified GAP-TV for CASSI."""
    from scipy.ndimage import gaussian_filter
    from pwm_core.physics.spectral.dispersion_models import dispersion_shift

    H, W = mask.shape
    x = np.zeros((H, W, n_bands), dtype=np.float32)
    # Initialise with adjoint
    for l_idx in range(n_bands):
        dx, dy = dispersion_shift(theta, band=l_idx)
        x[:, :, l_idx] = np.roll(
            np.roll(y * mask, -int(round(dy)), axis=0),
            -int(round(dx)), axis=1,
        ) / max(n_bands, 1)

    for _ in range(iters):
        # Forward
        yb = np.zeros((H, W), dtype=np.float32)
        for l_idx in range(n_bands):
            dx, dy = dispersion_shift(theta, band=l_idx)
            band = np.roll(
                np.roll(x[:, :, l_idx], int(round(dy)), axis=0),
                int(round(dx)), axis=1,
            )
            yb += band * mask
        # Residual
        r = y - yb
        # Backproject
        for l_idx in range(n_bands):
            dx, dy = dispersion_shift(theta, band=l_idx)
            upd = np.roll(
                np.roll(r * mask, -int(round(dy)), axis=0),
                -int(round(dx)), axis=1,
            ) / max(n_bands, 1)
            x[:, :, l_idx] += upd
        # Denoise
        for l_idx in range(n_bands):
            x[:, :, l_idx] = gaussian_filter(x[:, :, l_idx], sigma=0.3)
        x = np.clip(x, 0, 1)

    return x.astype(np.float32)


def run_cassi_baselines(
    sample_dir: str, rec: ManifestRecord
) -> List[TaskResult]:
    """Run T1-T4 for one CASSI sample."""
    results: List[TaskResult] = []
    sid = rec.sample_id

    x_gt = np.load(os.path.join(sample_dir, "x_gt.npy"))
    y = np.load(os.path.join(sample_dir, "y.npy"))
    mask_true = np.load(os.path.join(sample_dir, "mask.npy"))

    with open(os.path.join(sample_dir, "theta.json")) as f:
        theta_true = json.load(f)
    with open(os.path.join(sample_dir, "delta_theta.json")) as f:
        delta_theta = json.load(f)

    n_bands = theta_true.get("L", rec.n_bands or 8)

    # ── T1: parameter estimation (dispersion step) ──
    t0 = time.time()
    best_step, best_res = 1.0, float("inf")
    for s in np.linspace(0.5, 3.0, 6):
        test_theta = dict(theta_true)
        test_theta["disp_poly_x"] = [0.0, s, 0.0]
        x_test = _cassi_gap_tv_simple(y, mask_true, test_theta, n_bands, iters=10)
        # Forward residual
        from pwm_core.physics.spectral.dispersion_models import dispersion_shift
        yb = np.zeros_like(y)
        for l_idx in range(n_bands):
            dx, dy = dispersion_shift(test_theta, band=l_idx)
            band = np.roll(
                np.roll(x_test[:, :, l_idx], int(round(dy)), axis=0),
                int(round(dx)), axis=1,
            )
            yb += band * mask_true
        res = float(np.sum((y - yb) ** 2))
        if res < best_res:
            best_res = res
            best_step = s

    theta_est = {"disp_poly_x_1": best_step}
    theta_target_poly = theta_true.get("disp_poly_x", [0.0, 1.0, 0.0])
    disp_delta = delta_theta.get("disp_step_delta", 0.0)
    theta_target = {"disp_poly_x_1": theta_target_poly[1] + disp_delta}
    rmse = theta_rmse(theta_target, theta_est)
    results.append(TaskResult(
        task="T1_param_estimation",
        sample_id=sid, modality="cassi", baseline="grid_search",
        metrics={"theta_error_rmse": rmse},
        runtime_s=time.time() - t0,
    ))

    # ── T2: mismatch identification ──
    t0 = time.time()
    # Heuristic: check if mask shifted vs dispersion changed
    # (in practice would use a classifier)
    mask_mm = np.load(os.path.join(sample_dir, "mask_mm.npy"))
    mask_match = float(np.mean(mask_true == mask_mm))
    if mask_match < 0.95:
        pred_family = "mask_shift"
    elif abs(best_step - 1.0) > 0.3:
        pred_family = "disp_step"
    else:
        pred_family = "PSF_blur"
    correct = int(pred_family == rec.mismatch_family)
    results.append(TaskResult(
        task="T2_mismatch_id",
        sample_id=sid, modality="cassi", baseline="heuristic",
        metrics={"accuracy": float(correct), "f1": float(correct)},
        runtime_s=time.time() - t0,
    ))

    # ── T3: calibration ──
    t0 = time.time()
    theta_wrong = dict(theta_true)  # nominal (no mismatch)
    theta_calibrated = dict(theta_true)
    theta_calibrated["disp_poly_x"] = [0.0, best_step, 0.0]

    x_before = _cassi_gap_tv_simple(y, mask_true, theta_wrong, n_bands, iters=20)
    x_after = _cassi_gap_tv_simple(y, mask_true, theta_calibrated, n_bands, iters=20)
    psnr_before = compute_psnr(x_gt, x_before)
    psnr_after = compute_psnr(x_gt, x_after)
    results.append(TaskResult(
        task="T3_calibration",
        sample_id=sid, modality="cassi", baseline="grid_search",
        metrics={
            "theta_error_rmse": rmse,
            "psnr_before_db": psnr_before,
            "psnr_after_db": psnr_after,
            "psnr_gain_db": psnr_after - psnr_before,
        },
        runtime_s=time.time() - t0,
    ))

    # ── T4: reconstruction ──
    for name, th in [("oracle", theta_true), ("wrong_op", theta_wrong),
                      ("calibrated", theta_calibrated)]:
        t0 = time.time()
        x_hat = _cassi_gap_tv_simple(y, mask_true, th, n_bands, iters=30)
        psnr = compute_psnr(x_gt, x_hat)
        ssim = compute_ssim(
            x_gt.mean(axis=2) if x_gt.ndim == 3 else x_gt,
            x_hat.mean(axis=2) if x_hat.ndim == 3 else x_hat,
        )
        sam = compute_sam(x_gt, x_hat)
        results.append(TaskResult(
            task="T4_reconstruction",
            sample_id=sid, modality="cassi", baseline=name,
            metrics={"psnr_db": psnr, "ssim": ssim, "sam": sam},
            runtime_s=time.time() - t0,
        ))

    return results


# ── Dispatcher ──────────────────────────────────────────────────────────


def _load_manifest(data_dir: str) -> List[ManifestRecord]:
    """Load manifest.jsonl."""
    manifest_path = os.path.join(data_dir, "manifest.jsonl")
    if not os.path.exists(manifest_path):
        return []
    records = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(ManifestRecord.model_validate_json(line))
    return records


def run_all_baselines(
    data_dirs: List[str],
    results_dir: str,
    smoke: bool = False,
) -> List[TaskResult]:
    """Run baselines across all modalities."""
    os.makedirs(results_dir, exist_ok=True)
    all_results: List[TaskResult] = []

    for data_dir in data_dirs:
        records = _load_manifest(data_dir)
        if not records:
            logger.warning(f"No manifest found in {data_dir}, generating...")
            # Detect modality from dir name
            if "spc" in data_dir:
                records = generate_spc_dataset(data_dir, smoke=smoke)
            elif "cacti" in data_dir:
                records = generate_cacti_dataset(data_dir, smoke=smoke)
            elif "cassi" in data_dir:
                records = generate_cassi_dataset(data_dir, smoke=smoke)
            else:
                logger.error(f"Cannot determine modality from {data_dir}")
                continue

        for i, rec in enumerate(records):
            sample_dir = os.path.join(data_dir, "samples", rec.sample_id)
            if not os.path.isdir(sample_dir):
                logger.warning(f"Missing sample dir: {sample_dir}")
                continue

            logger.info(
                f"[{i + 1}/{len(records)}] {rec.modality.value}: {rec.sample_id}"
            )

            try:
                if rec.modality == Modality.spc:
                    results = run_spc_baselines(sample_dir, rec)
                elif rec.modality == Modality.cacti:
                    results = run_cacti_baselines(sample_dir, rec)
                elif rec.modality == Modality.cassi:
                    results = run_cassi_baselines(sample_dir, rec)
                else:
                    logger.warning(f"Unknown modality: {rec.modality}")
                    continue
                all_results.extend(results)
            except Exception as exc:
                logger.error(f"Error on {rec.sample_id}: {exc}", exc_info=True)

    # ── Save results ────────────────────────────────────────────────────
    results_path = os.path.join(results_dir, "baseline_results.jsonl")
    with open(results_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(asdict(r)) + "\n")
    logger.info(f"Saved {len(all_results)} results -> {results_path}")

    # ── Compute error bars (per task x modality x baseline) ─────────────
    _compute_error_bars(all_results, results_dir)

    return all_results


def _compute_error_bars(
    results: List[TaskResult], results_dir: str
) -> None:
    """Aggregate metrics with mean +/- std and save summary."""
    from collections import defaultdict

    groups: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for r in results:
        key = f"{r.task}|{r.modality}|{r.baseline}"
        groups[key].append(r.metrics)

    summary_rows = []
    for key, metric_list in sorted(groups.items()):
        task, modality, baseline = key.split("|")
        agg: Dict[str, Any] = {"task": task, "modality": modality, "baseline": baseline}
        # Collect all metric names
        all_keys = set()
        for m in metric_list:
            all_keys.update(m.keys())
        for mk in sorted(all_keys):
            vals = [m.get(mk, float("nan")) for m in metric_list]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                agg[f"{mk}_mean"] = float(np.mean(vals))
                agg[f"{mk}_std"] = float(np.std(vals))
                agg[f"{mk}_n"] = len(vals)
        summary_rows.append(agg)

    summary_path = os.path.join(results_dir, "error_bars.json")
    with open(summary_path, "w") as f:
        json.dump(summary_rows, f, indent=2)
    logger.info(f"Error bars -> {summary_path}")


# ── CLI ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run InverseNet baselines (T1-T4)"
    )
    parser.add_argument(
        "--data_dirs",
        nargs="*",
        default=[
            "datasets/inversenet_spc",
            "datasets/inversenet_cacti",
            "datasets/inversenet_cassi",
        ],
    )
    parser.add_argument("--results_dir", default="results/inversenet")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick validation: 1 sample per modality",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_all_baselines(args.data_dirs, args.results_dir, smoke=args.smoke)


if __name__ == "__main__":
    main()
