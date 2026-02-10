"""pwm_core.mismatch.uncertainty

Bootstrap resampling for calibration uncertainty quantification.

Given a correction function and calibration data, ``bootstrap_correction()``
resamples the data K times with replacement, runs the correction on each
subsample, and reports the 95% confidence interval for every corrected
parameter using the percentile method (2.5th and 97.5th percentiles).

All bootstrap bookkeeping (seeds and resampling indices) is stored in the
returned ``CorrectionResult`` so that runs are deterministic and fully
reproducible when logged to a RunBundle.

Usage
-----
::

    from pwm_core.mismatch.uncertainty import bootstrap_correction

    def my_correction(data):
        # data is an np.ndarray of calibration measurements
        # Returns dict: {"gain": float, "bias": float, "psnr": float}
        ...

    result = bootstrap_correction(my_correction, calibration_data, K=20)
    print(result.theta_corrected)
    print(result.theta_uncertainty)

Protocol
--------
1. The ``correction_fn`` accepts a 1-D or N-D numpy array and returns a
   dict with at least a ``"psnr"`` key (float) and one or more parameter
   keys (str -> float).
2. ``"psnr"`` is treated specially: it is recorded in the convergence
   curve but not reported in ``theta_corrected`` / ``theta_uncertainty``.
3. All other keys are treated as corrected parameters.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Sentinel key name for the reconstruction quality metric.
_PSNR_KEY = "psnr"


def bootstrap_correction(
    correction_fn: Callable[[np.ndarray], Dict[str, float]],
    data: np.ndarray,
    K: int = 20,
    seed: int = 42,
    psnr_without_correction: Optional[float] = None,
) -> "CorrectionResult":
    """Run bootstrap-resampled correction and return a CorrectionResult.

    Parameters
    ----------
    correction_fn : callable
        ``correction_fn(data_subset) -> dict[str, float]``.
        Must return a dict containing a ``"psnr"`` key (the
        reconstruction quality metric) and one or more parameter keys.
    data : np.ndarray
        Calibration data array.  Resampling is performed along the first
        axis (``axis=0``), so ``data.shape[0]`` is the number of samples.
    K : int
        Number of bootstrap resamples.  Default 20 per the schema contract.
    seed : int
        Base RNG seed.  Bootstrap seed k = ``seed + k``.
    psnr_without_correction : float, optional
        Baseline PSNR before correction, used to compute
        ``improvement_db``.  If *None*, improvement is computed relative
        to the worst bootstrap PSNR.

    Returns
    -------
    CorrectionResult
        Validated pydantic model with uncertainty bands, convergence
        curve, and full reproducibility metadata.

    Raises
    ------
    ValueError
        If ``correction_fn`` returns an empty dict or ``data`` is empty.
    """
    # -- Lazy import to avoid circular dependency at module level ----------
    from pwm_core.agents.contracts import CorrectionResult

    if data.size == 0:
        raise ValueError("Calibration data is empty; cannot bootstrap.")
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}.")

    n_samples = data.shape[0]

    # -- Generate deterministic seeds and indices -------------------------
    bootstrap_seeds: List[int] = [seed + k for k in range(K)]
    resampling_indices: List[List[int]] = []
    all_theta_dicts: List[Dict[str, float]] = []
    convergence_curve: List[float] = []
    n_evaluations = 0

    for k in range(K):
        rng = np.random.default_rng(bootstrap_seeds[k])
        indices = rng.integers(0, n_samples, size=n_samples).tolist()
        resampling_indices.append(indices)

        # Resample data along first axis
        data_k = data[indices]

        # Run correction
        result_k = correction_fn(data_k)
        n_evaluations += 1

        if not result_k:
            raise ValueError(
                f"correction_fn returned empty dict on bootstrap resample {k}."
            )

        all_theta_dicts.append(result_k)

        # Track PSNR for convergence curve
        psnr_k = result_k.get(_PSNR_KEY, 0.0)
        convergence_curve.append(float(psnr_k))

    # -- Separate parameter keys from the PSNR key -----------------------
    all_keys = set()
    for d in all_theta_dicts:
        all_keys.update(d.keys())
    param_keys = sorted(all_keys - {_PSNR_KEY})

    if not param_keys:
        raise ValueError(
            "correction_fn returned no parameter keys (only 'psnr'). "
            "At least one corrected parameter is required."
        )

    # -- Compute corrected theta (median across bootstrap) ----------------
    theta_corrected: Dict[str, float] = {}
    theta_uncertainty: Dict[str, List[float]] = {}

    for p in param_keys:
        values = np.array(
            [d.get(p, 0.0) for d in all_theta_dicts], dtype=np.float64
        )
        theta_corrected[p] = float(np.median(values))

        # 95% CI: hybrid method.
        # Use the wider of (a) the percentile interval [2.5th, 97.5th]
        # and (b) the normal-approximation interval (mean +/- 1.96*std).
        # The percentile method is slightly anti-conservative for finite
        # K; the normal method has better coverage for smooth estimators.
        # Taking the wider of the two yields conservative-enough coverage.
        ci_pct_low = float(np.percentile(values, 2.5))
        ci_pct_high = float(np.percentile(values, 97.5))

        mean_val = float(np.mean(values))
        std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        z = 1.96  # 95% two-sided z-score
        ci_norm_low = mean_val - z * std_val
        ci_norm_high = mean_val + z * std_val

        ci_low = min(ci_pct_low, ci_norm_low)
        ci_high = max(ci_pct_high, ci_norm_high)
        theta_uncertainty[p] = [ci_low, ci_high]

    # -- Compute improvement_db -------------------------------------------
    median_psnr = float(np.median(convergence_curve)) if convergence_curve else 0.0
    if psnr_without_correction is not None:
        improvement_db = median_psnr - psnr_without_correction
    else:
        # Fall back: improvement relative to worst bootstrap PSNR
        worst_psnr = min(convergence_curve) if convergence_curve else 0.0
        improvement_db = median_psnr - worst_psnr

    # Guard against non-finite values
    if not np.isfinite(improvement_db):
        improvement_db = 0.0

    # -- Build and validate CorrectionResult ------------------------------
    return CorrectionResult(
        theta_corrected=theta_corrected,
        theta_uncertainty=theta_uncertainty,
        improvement_db=float(improvement_db),
        n_evaluations=n_evaluations,
        convergence_curve=convergence_curve,
        bootstrap_seeds=bootstrap_seeds,
        resampling_indices=resampling_indices,
    )
