"""InverseNet manifest schema.

Each line of ``manifest.jsonl`` describes one sample in the dataset.
This module provides the Pydantic model for validation and I/O.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Modality(str, Enum):
    spc = "spc"
    cacti = "cacti"
    cassi = "cassi"


class Severity(str, Enum):
    mild = "mild"
    moderate = "moderate"
    severe = "severe"


class ManifestRecord(BaseModel):
    """Schema for one row of manifest.jsonl."""

    # --- identifiers ---
    sample_id: str = Field(..., description="Unique sample identifier")
    modality: Modality
    seed: int

    # --- physics configuration ---
    photon_level: float = Field(..., description="Mean photon count (e.g. 1e3)")
    compression_ratio: Optional[float] = Field(
        default=None,
        description="For SPC: measurement ratio; for CACTI: n_frames; for CASSI: n_bands",
    )
    n_frames: Optional[int] = Field(default=None, description="CACTI frame count")
    n_bands: Optional[int] = Field(default=None, description="CASSI spectral bands")

    # --- mismatch ---
    mismatch_family: str = Field(..., description="e.g. gain, mask_error, disp_step")
    severity: Severity

    # --- ground truth parameters ---
    theta: Dict[str, Any] = Field(default_factory=dict, description="True operator params")
    delta_theta: Dict[str, Any] = Field(
        default_factory=dict, description="Applied mismatch delta"
    )

    # --- file paths (relative to dataset root) ---
    paths: Dict[str, str] = Field(
        default_factory=dict,
        description="Artifact paths: x_gt, y, mask, theta, delta_theta, y_cal, ...",
    )

    # --- provenance ---
    git_hash: Optional[str] = None
    pwm_version: Optional[str] = None

    model_config = ConfigDict(extra="forbid")
