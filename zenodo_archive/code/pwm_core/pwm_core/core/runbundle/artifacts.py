"""pwm_core.core.runbundle.artifacts

Save reconstruction artifacts to RunBundle.

Artifacts include:
- artifacts/x_hat.npy - reconstructed signal
- artifacts/y.npy - measurements
- artifacts/x_true.npy - ground truth (if available)
- artifacts/metrics.json - quality metrics
- artifacts/images/ - PNG visualizations
- internal_state/diagnosis.json - diagnosis result
- internal_state/recon_info.json - reconstruction metadata
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from pwm_core.api.types import DiagnosisResult, ReconResult


def save_array(path: str, arr: np.ndarray) -> None:
    """Save numpy array to .npy file."""
    np.save(path, arr)


def save_json(path: str, data: Dict[str, Any]) -> None:
    """Save dictionary to JSON file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)


def _prepare_for_display(arr: np.ndarray) -> np.ndarray:
    """Prepare an array for image display.

    Handles:
    - Complex arrays (show magnitude)
    - 3D arrays (select middle slice)
    - 4D+ arrays (select middle slices)
    - 1D arrays (reshape to 2D if possible)

    Returns:
        2D real-valued array suitable for display.
    """
    # Handle complex arrays
    if np.iscomplexobj(arr):
        arr = np.abs(arr)

    # Handle different dimensionalities
    if arr.ndim == 1:
        # 1D: try to reshape to square, otherwise show as row
        n = len(arr)
        sqrt_n = int(np.sqrt(n))
        if sqrt_n * sqrt_n == n:
            arr = arr.reshape(sqrt_n, sqrt_n)
        else:
            # Show as a thin horizontal strip
            arr = arr.reshape(1, -1)
    elif arr.ndim == 3:
        # 3D: select middle slice along last axis
        mid = arr.shape[2] // 2
        arr = arr[:, :, mid]
    elif arr.ndim == 4:
        # 4D: select middle slices
        mid2 = arr.shape[2] // 2
        mid3 = arr.shape[3] // 2
        arr = arr[:, :, mid2, mid3]
    elif arr.ndim > 4:
        # Higher dims: flatten to 2D
        arr = arr.reshape(arr.shape[0], -1)

    return arr.astype(np.float32)


def _normalize_for_display(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] range for display."""
    arr = arr.astype(np.float32)
    vmin, vmax = np.percentile(arr, [1, 99])
    if vmax - vmin < 1e-8:
        vmax = vmin + 1.0
    arr = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    return arr


def save_image(path: str, arr: np.ndarray, title: str = "") -> None:
    """Save a numpy array as a PNG image.

    Args:
        path: Output file path.
        arr: Array to visualize.
        title: Optional title for the image.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        # Prepare array for display
        display_arr = _prepare_for_display(arr)
        display_arr = _normalize_for_display(display_arr)

        # Create figure
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(display_arr, cmap='gray', aspect='auto')
        if title:
            ax.set_title(title)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Save
        fig.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

    except ImportError:
        # matplotlib not available, skip image saving
        pass
    except Exception:
        # Don't fail the whole pipeline if image saving fails
        pass


def save_comparison_image(
    path: str,
    x_true: Optional[np.ndarray],
    x_hat: np.ndarray,
    y: np.ndarray,
) -> None:
    """Save a side-by-side comparison image.

    Args:
        path: Output file path.
        x_true: Ground truth (optional).
        x_hat: Reconstruction.
        y: Measurements.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Prepare arrays
        x_hat_disp = _normalize_for_display(_prepare_for_display(x_hat))
        y_disp = _normalize_for_display(_prepare_for_display(y))

        if x_true is not None:
            x_true_disp = _normalize_for_display(_prepare_for_display(x_true))

            # 3-panel comparison
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(x_true_disp, cmap='gray', aspect='auto')
            axes[0].set_title('Ground Truth')
            axes[0].axis('off')

            axes[1].imshow(x_hat_disp, cmap='gray', aspect='auto')
            axes[1].set_title('Reconstruction')
            axes[1].axis('off')

            axes[2].imshow(y_disp, cmap='gray', aspect='auto')
            axes[2].set_title('Measurement')
            axes[2].axis('off')
        else:
            # 2-panel comparison
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            axes[0].imshow(x_hat_disp, cmap='gray', aspect='auto')
            axes[0].set_title('Reconstruction')
            axes[0].axis('off')

            axes[1].imshow(y_disp, cmap='gray', aspect='auto')
            axes[1].set_title('Measurement')
            axes[1].axis('off')

        plt.tight_layout()
        fig.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

    except ImportError:
        pass
    except Exception:
        pass


def save_images(
    images_dir: str,
    x_hat: np.ndarray,
    y: np.ndarray,
    x_true: Optional[np.ndarray] = None,
) -> Dict[str, str]:
    """Save PNG visualizations of arrays.

    Args:
        images_dir: Directory to save images.
        x_hat: Reconstructed signal.
        y: Measurements.
        x_true: Optional ground truth.

    Returns:
        Dictionary mapping image names to file paths.
    """
    Path(images_dir).mkdir(parents=True, exist_ok=True)
    saved = {}

    # Save individual images
    x_hat_path = os.path.join(images_dir, "x_hat.png")
    save_image(x_hat_path, x_hat, "Reconstruction")
    saved["x_hat_png"] = x_hat_path

    y_path = os.path.join(images_dir, "y.png")
    save_image(y_path, y, "Measurement")
    saved["y_png"] = y_path

    if x_true is not None:
        x_true_path = os.path.join(images_dir, "x_true.png")
        save_image(x_true_path, x_true, "Ground Truth")
        saved["x_true_png"] = x_true_path

    # Save comparison image
    comparison_path = os.path.join(images_dir, "comparison.png")
    save_comparison_image(comparison_path, x_true, x_hat, y)
    saved["comparison_png"] = comparison_path

    return saved


def save_artifacts(
    rb_dir: str,
    x_hat: np.ndarray,
    y: np.ndarray,
    metrics: Dict[str, Any],
    diagnosis: Optional[DiagnosisResult] = None,
    x_true: Optional[np.ndarray] = None,
    recon_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Save all reconstruction artifacts to RunBundle directory.

    Args:
        rb_dir: RunBundle directory path.
        x_hat: Reconstructed signal.
        y: Measurements.
        metrics: Quality metrics dictionary.
        diagnosis: Optional diagnosis result.
        x_true: Optional ground truth signal.
        recon_info: Optional reconstruction metadata.

    Returns:
        Dictionary mapping artifact names to file paths.
    """
    artifacts_dir = os.path.join(rb_dir, "artifacts")
    internal_dir = os.path.join(rb_dir, "internal_state")
    images_dir = os.path.join(artifacts_dir, "images")

    # Ensure directories exist
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    Path(internal_dir).mkdir(parents=True, exist_ok=True)
    Path(images_dir).mkdir(parents=True, exist_ok=True)

    saved = {}

    # Save reconstructed signal
    x_hat_path = os.path.join(artifacts_dir, "x_hat.npy")
    save_array(x_hat_path, x_hat)
    saved["x_hat"] = x_hat_path

    # Save measurements
    y_path = os.path.join(artifacts_dir, "y.npy")
    save_array(y_path, y)
    saved["y"] = y_path

    # Save ground truth if available
    if x_true is not None:
        x_true_path = os.path.join(artifacts_dir, "x_true.npy")
        save_array(x_true_path, x_true)
        saved["x_true"] = x_true_path

    # Save metrics
    metrics_path = os.path.join(artifacts_dir, "metrics.json")
    save_json(metrics_path, metrics)
    saved["metrics"] = metrics_path

    # Save PNG images
    image_paths = save_images(images_dir, x_hat, y, x_true)
    saved.update(image_paths)

    # Save diagnosis
    if diagnosis is not None:
        diag_path = os.path.join(internal_dir, "diagnosis.json")
        diag_dict = _diagnosis_to_dict(diagnosis)
        save_json(diag_path, diag_dict)
        saved["diagnosis"] = diag_path

    # Save recon info
    if recon_info is not None:
        recon_info_path = os.path.join(internal_dir, "recon_info.json")
        save_json(recon_info_path, recon_info)
        saved["recon_info"] = recon_info_path

    return saved


def _diagnosis_to_dict(diagnosis: DiagnosisResult) -> Dict[str, Any]:
    """Convert DiagnosisResult to JSON-serializable dict."""
    result = {
        "verdict": diagnosis.verdict,
        "confidence": diagnosis.confidence,
        "bottleneck": diagnosis.bottleneck,
        "evidence": diagnosis.evidence,
        "suggested_actions": [],
    }

    for action in diagnosis.suggested_actions:
        action_dict = {
            "knob": action.knob,
            "op": action.op.value if hasattr(action.op, 'value') else str(action.op),
            "val": action.val,
            "reason": action.reason,
        }
        result["suggested_actions"].append(action_dict)

    return result


def save_trace(rb_dir: str, trace: Dict[str, np.ndarray]) -> Dict[str, str]:
    """Save node-by-node trace arrays and PNG visualizations.

    Args:
        rb_dir: RunBundle directory path.
        trace: Dict mapping "{idx:02d}_{node_id}" to numpy arrays.

    Returns:
        Dict mapping trace keys to saved file paths.
    """
    trace_dir = os.path.join(rb_dir, "artifacts", "trace")
    png_dir = os.path.join(trace_dir, "png")
    Path(trace_dir).mkdir(parents=True, exist_ok=True)
    Path(png_dir).mkdir(parents=True, exist_ok=True)

    saved = {}
    for key, arr in trace.items():
        # Save .npy
        npy_path = os.path.join(trace_dir, f"{key}.npy")
        save_array(npy_path, arr)
        saved[f"trace_{key}_npy"] = npy_path

        # Save .png visualization
        png_path = os.path.join(png_dir, f"{key}.png")
        _save_trace_png(png_path, arr, title=key)
        saved[f"trace_{key}_png"] = png_path

    return saved


def _save_trace_png(path: str, arr: np.ndarray, title: str = "") -> None:
    """Save trace array as PNG with appropriate visualization.

    Rules:
    - 2D array -> grayscale image
    - 3D array (H,W,C) -> RGB or first 3 channels
    - 1D array -> line plot
    - Complex array -> magnitude + phase side-by-side
    - 4D+ -> slice through first 2 spatial dims
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if np.iscomplexobj(arr):
            # Magnitude + phase side-by-side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            mag = _normalize_for_display(_prepare_for_display(np.abs(arr)))
            phase = _prepare_for_display(np.angle(arr))
            ax1.imshow(mag, cmap='gray', aspect='auto')
            ax1.set_title(f'{title} (magnitude)')
            ax1.axis('off')
            ax2.imshow(phase, cmap='twilight', aspect='auto')
            ax2.set_title(f'{title} (phase)')
            ax2.axis('off')
        elif arr.ndim == 1:
            # Line plot
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(arr[:min(len(arr), 2000)])
            ax.set_title(title)
            ax.set_xlabel('index')
            ax.set_ylabel('value')
        else:
            # 2D/3D/4D+ -> image
            fig, ax = plt.subplots(figsize=(6, 6))
            display_arr = _normalize_for_display(_prepare_for_display(arr))
            ax.imshow(display_arr, cmap='gray', aspect='auto')
            ax.set_title(title)
            ax.axis('off')

        fig.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
    except ImportError:
        pass
    except Exception:
        pass


def save_operator_meta(rb_dir: str, meta: Dict[str, Any]) -> str:
    """Save W2 operator metadata to RunBundle.

    Args:
        rb_dir: RunBundle directory path.
        meta: Operator metadata dict with keys:
            a_definition, a_extraction_method, a_shape, a_dtype,
            a_sha256, a_nnz, a_sparsity, linearity,
            linearization_notes, mismatch_type, mismatch_params,
            correction_family, nll_before, nll_after,
            nll_decrease_pct, timestamp

    Returns:
        Path to saved JSON file.
    """
    meta_path = os.path.join(rb_dir, "artifacts", "w2_operator_meta.json")
    save_json(meta_path, meta)
    return meta_path


def compute_operator_hash(operator) -> str:
    """Compute SHA-256 hash of an operator's data.

    Handles dense arrays, sparse matrices, and callables.

    Args:
        operator: numpy array, scipy sparse matrix, or callable.

    Returns:
        SHA-256 hex digest, or "N/A (callable)" for pure callables.
    """
    import hashlib

    if hasattr(operator, 'tobytes'):
        # Dense numpy array
        return hashlib.sha256(operator.tobytes()).hexdigest()
    elif hasattr(operator, 'data') and hasattr(operator, 'indices') and hasattr(operator, 'indptr'):
        # CSR/CSC sparse matrix
        h = hashlib.sha256()
        h.update(operator.data.tobytes())
        h.update(operator.indices.tobytes())
        h.update(operator.indptr.tobytes())
        return h.hexdigest()
    elif hasattr(operator, 'toarray'):
        # Other sparse format - convert to dense
        return hashlib.sha256(operator.toarray().tobytes()).hexdigest()
    else:
        return "N/A (callable)"


def load_artifacts(rb_dir: str) -> Dict[str, Any]:
    """Load artifacts from a RunBundle directory.

    Args:
        rb_dir: RunBundle directory path.

    Returns:
        Dictionary with loaded artifacts.
    """
    artifacts_dir = os.path.join(rb_dir, "artifacts")
    internal_dir = os.path.join(rb_dir, "internal_state")

    result = {}

    # Load arrays
    x_hat_path = os.path.join(artifacts_dir, "x_hat.npy")
    if os.path.exists(x_hat_path):
        result["x_hat"] = np.load(x_hat_path)

    y_path = os.path.join(artifacts_dir, "y.npy")
    if os.path.exists(y_path):
        result["y"] = np.load(y_path)

    x_true_path = os.path.join(artifacts_dir, "x_true.npy")
    if os.path.exists(x_true_path):
        result["x_true"] = np.load(x_true_path)

    # Load JSON files
    metrics_path = os.path.join(artifacts_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r', encoding='utf-8') as f:
            result["metrics"] = json.load(f)

    diag_path = os.path.join(internal_dir, "diagnosis.json")
    if os.path.exists(diag_path):
        with open(diag_path, 'r', encoding='utf-8') as f:
            result["diagnosis"] = json.load(f)

    recon_info_path = os.path.join(internal_dir, "recon_info.json")
    if os.path.exists(recon_info_path):
        with open(recon_info_path, 'r', encoding='utf-8') as f:
            result["recon_info"] = json.load(f)

    return result


def create_recon_result(
    solver_id: str,
    rb_dir: str,
    metrics: Dict[str, Any],
    runtime_s: Optional[float] = None,
) -> ReconResult:
    """Create a ReconResult pointing to artifacts in RunBundle.

    Args:
        solver_id: Identifier of the solver used.
        rb_dir: RunBundle directory path.
        metrics: Quality metrics.
        runtime_s: Optional reconstruction runtime in seconds.

    Returns:
        ReconResult with references to artifacts.
    """
    return ReconResult(
        solver_id=solver_id,
        xhat_ref=os.path.join(rb_dir, "artifacts", "x_hat.npy"),
        yhat_ref=None,  # Could add y_hat if we compute forward of x_hat
        metrics=metrics,
        runtime_s=runtime_s,
    )
