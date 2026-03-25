"""pwm_core.cards.claim_card

ClaimCard v1 — links an arXiv paper claim to a benchmark result.

ClaimCards are the entry point for the arXiv scaffolder (P1 growth item).
A ClaimCard starts as a Draft and is promoted through the trust ladder as
the claim is verified against the benchmark.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    from dataclasses import dataclass as BaseModel  # type: ignore
    Field = lambda *a, **kw: None  # type: ignore


class ClaimCard(BaseModel):
    """Card linking an arXiv paper claim to a PWM benchmark result.

    Fields
    ------
    claim_id : str
        Unique identifier (``arxiv_{arxiv_id}_{claim_index}``).
    arxiv_id : str
        arXiv paper identifier (e.g. ``"2401.12345"``).
    title : str
        Paper title.
    claimed_metric : str
        The specific metric claimed (e.g. ``"PSNR"``).
    claimed_value : float, optional
        Claimed metric value.
    modality : str
        Modality that the claim applies to.
    method_id : str, optional
        Reference to the MethodCard describing the claimed algorithm.
    spec_id : str, optional
        Reference to the SpecCard describing the evaluation protocol.
    trust_tier : str
        Current trust tier (``"draft"`` initially).
    run_id : str, optional
        RunBundle that verified this claim.
    verified_value : float, optional
        Metric value measured by the harness (set after verification).
    authors : list[str]
    tags : list[str]
    created_at : str
    extra : dict, optional
    """

    claim_id: str
    arxiv_id: str
    title: str = ""
    claimed_metric: str = "PSNR"
    claimed_value: Optional[float] = None
    modality: str = ""
    method_id: Optional[str] = None
    spec_id: Optional[str] = None
    trust_tier: str = "draft"
    run_id: Optional[str] = None
    verified_value: Optional[float] = None
    authors: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    extra: Optional[Dict[str, Any]] = None
