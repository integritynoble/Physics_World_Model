"""Source attribution tracking for benchmark data and models.

Every benchmark result records where its ground truth, forward model,
solver, and mismatch ranges came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Optional


class SourceType(str, Enum):
    """How a benchmark component was sourced."""
    web = "web"               # Downloaded from a public URL / dataset repo
    paper = "paper"           # Parameters taken from a published paper
    experimental = "experimental"  # Real lab / clinical measurements
    synthetic_web = "synthetic_web"  # Synthetic data from an online generator
    generated = "generated"   # Programmatically generated (last resort)
    registry = "registry"     # From PWM contrib registries


@dataclass
class SourceRef:
    """A single source reference."""
    type: SourceType
    reference: str                 # Citation, URL, or description
    url: Optional[str] = None     # Direct URL if applicable
    license: Optional[str] = None

    def to_dict(self) -> Dict:
        d = {"type": self.type.value, "reference": self.reference}
        if self.url:
            d["url"] = self.url
        if self.license:
            d["license"] = self.license
        return d


@dataclass
class SourceAttribution:
    """Full source attribution for a benchmark run.

    Tracks provenance for ground truth data, the forward model,
    the reconstruction solver, and mismatch parameter ranges.
    """
    ground_truth: Optional[SourceRef] = None
    forward_model: Optional[SourceRef] = None
    solver: Optional[SourceRef] = None
    mismatch_ranges: Optional[SourceRef] = None
    extra: Dict[str, SourceRef] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = {}
        if self.ground_truth:
            d["ground_truth"] = self.ground_truth.to_dict()
        if self.forward_model:
            d["forward_model"] = self.forward_model.to_dict()
        if self.solver:
            d["solver"] = self.solver.to_dict()
        if self.mismatch_ranges:
            d["mismatch_ranges"] = self.mismatch_ranges.to_dict()
        for k, v in self.extra.items():
            d[k] = v.to_dict()
        return d

    @classmethod
    def from_config(cls, cfg_dict: Dict) -> "SourceAttribution":
        """Build from a config's source_attribution section."""
        attr = cls()
        for key in ("ground_truth", "forward_model", "solver", "mismatch_ranges"):
            entry = cfg_dict.get(key)
            if entry:
                attr.__dict__[key] = SourceRef(
                    type=SourceType(entry["type"]),
                    reference=entry.get("reference", ""),
                    url=entry.get("url"),
                    license=entry.get("license"),
                )
        return attr
