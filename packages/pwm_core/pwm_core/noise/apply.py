"""Canonical noise application for PWM experiments.

Replaces 8+ copy-pasted ``_apply_photon_noise()`` functions with one
canonical version.  Default parameters reproduce **bit-identical** output
to the existing Poisson + Gaussian model.

Noise models
------------
- ``"poisson_gaussian"`` (default): Poisson shot noise + Gaussian read noise.
  Matches all existing experiment code.
- ``"poisson"``: Shot noise only (CT, ptychography).
- ``"gaussian"``: Thermal / read noise only (MRI).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def apply_photon_noise(
    y: np.ndarray,
    photon_level: float,
    rng: np.random.Generator,
    *,
    noise_model: str = "poisson_gaussian",
    read_sigma_fraction: float = 0.01,
) -> np.ndarray:
    """Apply physics-based noise to a measurement array.

    Parameters
    ----------
    y : ndarray
        Clean (or partially noisy) measurement.
    photon_level : float
        Target photon count at peak intensity.  For ``"gaussian"`` mode
        this is interpreted as an SNR proxy (sigma = 1/photon_level of
        signal range).
    rng : Generator
        NumPy random generator for reproducibility.
    noise_model : str
        One of ``"poisson_gaussian"``, ``"poisson"``, ``"gaussian"``.
    read_sigma_fraction : float
        Read-noise standard deviation as a fraction of
        ``sqrt(photon_level)``.  Only used by ``"poisson_gaussian"``.

    Returns
    -------
    ndarray
        Noisy measurement (same shape as *y*).
    """
    if noise_model == "poisson_gaussian":
        return _poisson_gaussian(y, photon_level, rng, read_sigma_fraction)
    elif noise_model == "poisson":
        return _poisson_only(y, photon_level, rng)
    elif noise_model == "gaussian":
        return _gaussian_only(y, photon_level, rng)
    else:
        raise ValueError(
            f"Unknown noise_model={noise_model!r}. "
            "Expected 'poisson_gaussian', 'poisson', or 'gaussian'."
        )


def _poisson_gaussian(
    y: np.ndarray,
    photon_level: float,
    rng: np.random.Generator,
    read_sigma_fraction: float,
) -> np.ndarray:
    """Poisson shot + Gaussian read noise (existing behavior)."""
    scale = photon_level / (np.abs(y).max() + 1e-10)
    y_scaled = np.maximum(y * scale, 0)
    y_noisy = rng.poisson(y_scaled).astype(np.float32)
    read_sigma = np.sqrt(photon_level) * read_sigma_fraction
    y_noisy += rng.normal(0, read_sigma, size=y.shape).astype(np.float32)
    y_noisy /= scale
    return y_noisy


def _poisson_only(
    y: np.ndarray,
    photon_level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Poisson shot noise only (CT, ptychography)."""
    scale = photon_level / (np.abs(y).max() + 1e-10)
    y_scaled = np.maximum(y * scale, 0)
    y_noisy = rng.poisson(y_scaled).astype(np.float32)
    y_noisy /= scale
    return y_noisy


def _gaussian_only(
    y: np.ndarray,
    photon_level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Gaussian thermal noise only (MRI).

    *photon_level* is interpreted as an SNR proxy: ``sigma`` is the
    reciprocal of ``photon_level`` scaled by the signal range.
    """
    signal_range = np.abs(y).max() + 1e-10
    sigma = signal_range / photon_level
    y_noisy = y + rng.normal(0, sigma, size=y.shape).astype(y.dtype)
    return y_noisy


def get_noise_recipe(
    modality_key: str,
    level_name: str,
    photon_db: Any,
) -> Dict[str, Any]:
    """Look up noise recipe from photon_db for a modality + level.

    Parameters
    ----------
    modality_key : str
        Registry key (e.g. ``"cassi"``).
    level_name : str
        One of ``"bright"``, ``"standard"``, ``"low_light"``.
    photon_db : PhotonDbFileYaml or dict
        Parsed photon database.

    Returns
    -------
    dict
        ``{"n_photons": float, "noise_model": str,
        "read_sigma_fraction": float}`` ready for :func:`apply_photon_noise`.
    """
    # Support both pydantic model and raw dict
    if hasattr(photon_db, "modalities"):
        modalities = photon_db.modalities
    else:
        modalities = photon_db.get("modalities", {})

    if modality_key not in modalities:
        raise KeyError(
            f"Modality '{modality_key}' not found in photon_db. "
            f"Available: {sorted(modalities.keys())}"
        )

    entry = modalities[modality_key]

    # Get photon_levels (pydantic model or dict)
    if hasattr(entry, "photon_levels"):
        levels = entry.photon_levels
    else:
        levels = entry.get("photon_levels") if isinstance(entry, dict) else None

    if levels is None:
        raise KeyError(
            f"Modality '{modality_key}' has no photon_levels defined."
        )

    if level_name not in levels:
        raise KeyError(
            f"Level '{level_name}' not found for modality '{modality_key}'. "
            f"Available: {sorted(levels.keys())}"
        )

    level = levels[level_name]

    # Extract fields (pydantic model or dict)
    if hasattr(level, "n_photons"):
        n_photons = level.n_photons
        read_sigma_fraction = level.read_sigma_fraction
    else:
        n_photons = level.get("n_photons")
        read_sigma_fraction = level.get("read_sigma_fraction", 0.01)

    # Get noise_model from the modality entry
    if hasattr(entry, "noise_model"):
        noise_model = entry.noise_model or "poisson_gaussian"
    else:
        noise_model = entry.get("noise_model", "poisson_gaussian") if isinstance(entry, dict) else "poisson_gaussian"

    return {
        "n_photons": n_photons,
        "noise_model": noise_model,
        "read_sigma_fraction": read_sigma_fraction,
    }
