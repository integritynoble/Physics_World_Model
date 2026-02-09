"""pwm_core.agents.hybrid

Hybrid modality support.

Some applications combine two modalities (e.g., Fluorescence + Photoacoustic).
This module recognises hybrid combinations and orchestrates parallel
reconstruction + fusion.

Entirely deterministic — no LLM required.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .contracts import StrictBaseModel
from .base import AgentContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class HybridModalitySpec(StrictBaseModel):
    """Specification for a hybrid modality combination."""

    primary: str
    secondary: str
    fusion_strategy: str  # "parallel_recon_then_fuse" | "joint_forward"
    shared_parameters: Dict[str, Any] = {}
    description: str = ""


class FusionResult(StrictBaseModel):
    """Result of fusing two modality reconstructions."""

    primary_modality: str
    secondary_modality: str
    fusion_strategy: str
    combined_psnr_db: Optional[float] = None
    weights: Dict[str, float] = {}
    description: str = ""


# ---------------------------------------------------------------------------
# Hybrid database
# ---------------------------------------------------------------------------

# Recognised hybrid combinations with default fusion strategies
HYBRID_DB: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("photoacoustic", "widefield"): {
        "fusion_strategy": "parallel_recon_then_fuse",
        "description": (
            "Photoacoustic for deep vascular imaging + "
            "widefield for surface fluorescence"
        ),
        "weights": {"primary": 0.6, "secondary": 0.4},
    },
    ("oct", "confocal"): {
        "fusion_strategy": "parallel_recon_then_fuse",
        "description": (
            "OCT for depth structure + confocal for lateral resolution"
        ),
        "weights": {"primary": 0.5, "secondary": 0.5},
    },
    ("fpm", "holography"): {
        "fusion_strategy": "joint_forward",
        "description": (
            "FPM synthetic aperture + holographic phase"
        ),
        "weights": {"primary": 0.7, "secondary": 0.3},
    },
    ("ct", "mri"): {
        "fusion_strategy": "parallel_recon_then_fuse",
        "description": (
            "CT for bone/density + MRI for soft tissue contrast"
        ),
        "weights": {"primary": 0.5, "secondary": 0.5},
    },
    ("lightsheet", "sim"): {
        "fusion_strategy": "parallel_recon_then_fuse",
        "description": (
            "Light-sheet for volumetric + SIM for super-resolution"
        ),
        "weights": {"primary": 0.4, "secondary": 0.6},
    },
}


# ---------------------------------------------------------------------------
# Hybrid manager
# ---------------------------------------------------------------------------

class HybridModalityManager:
    """Detect and manage hybrid modality combinations.

    Workflow:
    1. ``detect()`` checks if a pair of modalities is a known hybrid.
    2. ``build_spec()`` creates a HybridModalitySpec for the combination.
    3. ``fuse()`` combines two reconstructed images using the fusion strategy.
    """

    @staticmethod
    def detect(modality_a: str, modality_b: str) -> bool:
        """Check if two modalities form a known hybrid combination.

        Order-independent: checks both (a, b) and (b, a).
        """
        return (
            (modality_a, modality_b) in HYBRID_DB
            or (modality_b, modality_a) in HYBRID_DB
        )

    @staticmethod
    def build_spec(
        modality_a: str,
        modality_b: str,
        shared_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[HybridModalitySpec]:
        """Build a HybridModalitySpec for a known combination.

        Parameters
        ----------
        modality_a, modality_b : str
            The two modalities to combine.
        shared_params : dict, optional
            Shared parameters (e.g., sample, wavelength).

        Returns
        -------
        HybridModalitySpec or None
            Spec if recognised, None otherwise.
        """
        key: Optional[Tuple[str, str]] = None
        if (modality_a, modality_b) in HYBRID_DB:
            key = (modality_a, modality_b)
        elif (modality_b, modality_a) in HYBRID_DB:
            key = (modality_b, modality_a)

        if key is None:
            return None

        entry = HYBRID_DB[key]
        return HybridModalitySpec(
            primary=key[0],
            secondary=key[1],
            fusion_strategy=entry["fusion_strategy"],
            shared_parameters=shared_params or {},
            description=entry.get("description", ""),
        )

    @staticmethod
    def list_known_hybrids() -> List[HybridModalitySpec]:
        """List all known hybrid combinations."""
        specs = []
        for (primary, secondary), entry in HYBRID_DB.items():
            specs.append(HybridModalitySpec(
                primary=primary,
                secondary=secondary,
                fusion_strategy=entry["fusion_strategy"],
                description=entry.get("description", ""),
            ))
        return specs

    @staticmethod
    def fuse(
        recon_primary: np.ndarray,
        recon_secondary: np.ndarray,
        spec: HybridModalitySpec,
    ) -> Tuple[np.ndarray, FusionResult]:
        """Fuse two modality reconstructions.

        Parameters
        ----------
        recon_primary : np.ndarray
            Reconstruction from the primary modality.
        recon_secondary : np.ndarray
            Reconstruction from the secondary modality.
        spec : HybridModalitySpec
            Hybrid specification.

        Returns
        -------
        fused : np.ndarray
            Fused reconstruction.
        result : FusionResult
            Fusion metadata.
        """
        key = (spec.primary, spec.secondary)
        entry = HYBRID_DB.get(key, {})
        weights = entry.get("weights", {"primary": 0.5, "secondary": 0.5})
        w_p = weights["primary"]
        w_s = weights["secondary"]

        if spec.fusion_strategy == "parallel_recon_then_fuse":
            fused = _weighted_average_fusion(
                recon_primary, recon_secondary, w_p, w_s
            )
        elif spec.fusion_strategy == "joint_forward":
            # Joint forward model fusion — use wavelet-based for coherent modalities
            fused = _wavelet_fusion(recon_primary, recon_secondary, w_p, w_s)
        else:
            # Fallback: weighted average
            fused = _weighted_average_fusion(
                recon_primary, recon_secondary, w_p, w_s
            )

        result = FusionResult(
            primary_modality=spec.primary,
            secondary_modality=spec.secondary,
            fusion_strategy=spec.fusion_strategy,
            weights=weights,
            description=spec.description,
        )

        return fused, result


# ---------------------------------------------------------------------------
# Fusion strategies
# ---------------------------------------------------------------------------

def _weighted_average_fusion(
    img_a: np.ndarray,
    img_b: np.ndarray,
    w_a: float,
    w_b: float,
) -> np.ndarray:
    """Simple weighted average fusion.

    Handles shape mismatches by resizing the smaller image to match
    the larger one using nearest-neighbour interpolation.
    """
    a = img_a.astype(np.float64)
    b = img_b.astype(np.float64)

    # Handle shape mismatch: crop or pad to match
    if a.shape != b.shape:
        # Use the smaller shape
        min_shape = tuple(min(s1, s2) for s1, s2 in zip(a.shape, b.shape))
        slices = tuple(slice(0, s) for s in min_shape)
        a = a[slices]
        b = b[slices]

    fused = w_a * a + w_b * b
    return fused.astype(np.float32)


def _wavelet_fusion(
    img_a: np.ndarray,
    img_b: np.ndarray,
    w_a: float,
    w_b: float,
) -> np.ndarray:
    """Wavelet-based fusion using FFT decomposition.

    Splits each image into low-frequency (averaged) and high-frequency
    (max-selected) components using a simple frequency-domain filter.
    """
    a = img_a.astype(np.float64)
    b = img_b.astype(np.float64)

    # Handle shape mismatch
    if a.shape != b.shape:
        min_shape = tuple(min(s1, s2) for s1, s2 in zip(a.shape, b.shape))
        slices = tuple(slice(0, s) for s in min_shape)
        a = a[slices]
        b = b[slices]

    if a.ndim < 2:
        # Fallback for 1D
        return _weighted_average_fusion(img_a, img_b, w_a, w_b)

    # FFT-based decomposition (2D)
    A = np.fft.fft2(a)
    B = np.fft.fft2(b)

    h, w_dim = a.shape[:2]
    fy = np.fft.fftfreq(h)[:, np.newaxis]
    fx = np.fft.fftfreq(w_dim)[np.newaxis, :]
    freq_mag = np.sqrt(fy ** 2 + fx ** 2)

    # Low-pass cutoff
    cutoff = 0.1
    lp = (freq_mag < cutoff).astype(np.float64)
    hp = 1.0 - lp

    # Low-frequency: weighted average
    L_fused = w_a * (A * lp) + w_b * (B * lp)

    # High-frequency: select larger magnitude
    A_hp = A * hp
    B_hp = B * hp
    mask = np.abs(A_hp) >= np.abs(B_hp)
    H_fused = np.where(mask, A_hp, B_hp)

    fused = np.real(np.fft.ifft2(L_fused + H_fused))
    return fused.astype(np.float32)
