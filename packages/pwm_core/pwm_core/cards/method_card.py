"""pwm_core.cards.method_card

MethodCard v1 — describes the algorithm / solver submitted for evaluation.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    from dataclasses import dataclass as BaseModel  # type: ignore
    Field = lambda *a, **kw: None  # type: ignore


class MethodCard(BaseModel):
    """Card describing a reconstruction / solver algorithm.

    Fields
    ------
    method_id : str
        Unique identifier (``solver_name_vN``).
    name : str
        Human-readable algorithm name.
    version : str
        Semantic version.
    category : str
        ``"classical"`` | ``"deep_learning"`` | ``"hybrid"``.
    description : str
        Algorithm description.
    paper_url : str, optional
        DOI or arXiv URL of the reference paper.
    code_url : str, optional
        Repository URL.
    authors : list[str]
        Author names or ORCID URIs.
    compute_class : str
        ``"cpu"`` | ``"gpu_t4"`` | ``"gpu_a100"`` | ``"modal"``.
    declared_budget_s : float, optional
        Self-declared compute budget in seconds (used for S4 gate).
    tags : list[str]
    created_at : str
    """

    method_id: str
    name: str
    version: str = "1.0.0"
    category: str = "classical"
    description: str = ""
    paper_url: Optional[str] = None
    code_url: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    compute_class: str = "cpu"
    declared_budget_s: Optional[float] = None
    tags: List[str] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    extra: Optional[Dict[str, Any]] = None
