"""pwm_core.cards.dataset_card

DatasetCard v1 — describes the dataset used for benchmark evaluation.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    from dataclasses import dataclass as BaseModel  # type: ignore
    Field = lambda *a, **kw: None  # type: ignore


class DatasetCard(BaseModel):
    """Card describing a benchmark dataset.

    Fields
    ------
    dataset_id : str
        Unique identifier (``modality_split_vN``).
    modality : str
        Physics modality this dataset covers.
    version : str
        Semantic version.
    split : str
        ``"public"`` | ``"dev"`` | ``"hidden"``.
    n_samples : int
        Number of samples in this split.
    source : str
        Dataset origin (``"real"`` | ``"synthetic"`` | ``"simulated"``).
    license : str
        SPDX license identifier.
    gcs_path : str, optional
        GCS URI (``gs://pwm-benchmark-datasets/datasets/Benchmark/...``).
    description : str
    authors : list[str]
    tags : list[str]
    created_at : str
    """

    dataset_id: str
    modality: str
    version: str = "1.0.0"
    split: str = "public"
    n_samples: int = 0
    source: str = "synthetic"
    license: str = "CC-BY-4.0"
    gcs_path: Optional[str] = None
    description: str = ""
    authors: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    extra: Optional[Dict[str, Any]] = None
