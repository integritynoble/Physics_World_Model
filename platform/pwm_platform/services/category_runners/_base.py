"""Base protocol for category-specific simulation runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MethodResult:
    """Per-method metrics across three scenarios."""
    key: str
    display_name: str
    method_type: str        # "classical", "pnp", "deep"
    psnr_i: float
    ssim_i: float
    psnr_ii: float
    ssim_ii: float
    psnr_iii: float
    ssim_iii: float
    gap_i_ii: float
    recovery_ii_iii: float


class CategoryRunner:
    """Base class for category-specific simulation runners.

    Each subclass implements physics appropriate to its category_module:
    generate a phantom, apply a forward model, and produce calibrated baselines.
    """

    def generate_phantom(
        self, config: dict, rng: np.random.Generator,
    ) -> Tuple[np.ndarray, str, str]:
        """Generate a phantom image for this category.

        Returns:
            (array, display_name, colormap)
        """
        raise NotImplementedError

    def apply_forward_model(
        self, phantom: np.ndarray, config: dict, rng: np.random.Generator,
    ) -> Tuple[np.ndarray, str, str]:
        """Apply the forward model to produce a measurement.

        Returns:
            (measurement_array, title, colormap)
        """
        raise NotImplementedError

    def get_baselines(
        self, config: dict, rng: np.random.Generator,
    ) -> Tuple[List[MethodResult], Dict[str, str], str, str]:
        """Generate calibrated baseline results.

        Returns:
            (methods, scenario_labels, dataset_label, attribution)
        """
        raise NotImplementedError
