"""pwm_core.cards.spec_card

SpecCard v1 — describes the forward model and measurement process.

A SpecCard is the first required docking artifact.  It references a
``CoreSpec`` (ExperimentSpec) by ID and adds human-readable metadata so
that the platform can index and display the spec without executing it.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    from dataclasses import dataclass as BaseModel  # type: ignore
    Field = lambda *a, **kw: None  # type: ignore


class SpecCard(BaseModel):
    """Card describing a physics forward model and measurement protocol.

    Fields
    ------
    spec_id : str
        Unique identifier (``modality_variant_vN``).
    modality : str
        Physics modality (e.g. ``"ct"``, ``"cassi"``).
    version : str
        Semantic version of this spec (``"1.0.0"``).
    title : str
        Short human-readable title.
    description : str
        Paragraph description of the measurement process.
    forward_model : str
        Name of the forward operator (e.g. ``"radon_parallel"``).
    noise_model : str
        Noise model applied (e.g. ``"poisson"``, ``"gaussian"``).
    authors : list[str]
        Names or ORCID URIs of spec authors.
    license : str
        SPDX license identifier (e.g. ``"MIT"``).
    tags : list[str]
        Free-form tags for search (e.g. ``["medical", "x-ray"]``).
    created_at : str
        ISO-8601 creation timestamp.
    spec_ref : dict, optional
        Minimal serialised CoreSpec for programmatic use.
    """

    spec_id: str
    modality: str
    version: str = "1.0.0"
    title: str
    description: str = ""
    forward_model: str = ""
    noise_model: str = ""
    authors: List[str] = Field(default_factory=list)
    license: str = "MIT"
    tags: List[str] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    spec_ref: Optional[Dict[str, Any]] = None
