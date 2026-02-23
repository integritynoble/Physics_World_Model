"""BenchmarkConfig – YAML-driven dataclass for a single modality benchmark.

Each modality has one YAML config in ``benchmarks/configs/<modality_id>.yaml``.
This module loads, validates, and exposes those configs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class MismatchParam:
    """One mismatch parameter for perturbation injection."""
    name: str
    nominal: float
    range: Tuple[float, float]
    unit: str = ""
    grid_values: Optional[List[float]] = None

    @classmethod
    def from_dict(cls, d: Dict) -> "MismatchParam":
        rng = d.get("range", [0, 0])
        return cls(
            name=d["name"],
            nominal=float(d.get("nominal", 0)),
            range=(float(rng[0]), float(rng[1])),
            unit=d.get("unit", ""),
            grid_values=d.get("grid_values"),
        )


@dataclass
class SolverSpec:
    """One solver configuration."""
    id: str
    tier: str  # traditional_cpu | famous_dl | best_quality | small_gpu
    module: str = ""
    function: str = ""
    params: str = "0"
    gpu: bool = False
    cfg: Dict[str, Any] = field(default_factory=dict)
    reference: str = ""

    @classmethod
    def from_dict(cls, d: Dict, tier: str) -> "SolverSpec":
        return cls(
            id=d.get("name", tier),
            tier=tier,
            module=d.get("module", ""),
            function=d.get("function", ""),
            params=str(d.get("params", "0")),
            gpu=d.get("gpu", False),
            cfg=d.get("cfg", {}),
            reference=d.get("reference", ""),
        )


@dataclass
class DataSourceSpec:
    """Data acquisition specification."""
    priority: List[str] = field(
        default_factory=lambda: ["web", "experimental", "synthetic_web", "generated"]
    )
    dataset_id: str = ""
    dataset_url: str = ""
    fallback: str = "generated"
    synthetic_generator: str = ""  # category module name
    citation: str = ""
    license: str = ""


@dataclass
class SourceAttributionSpec:
    """Source tracking entries as written in the config."""
    ground_truth: Dict[str, str] = field(default_factory=dict)
    forward_model: Dict[str, str] = field(default_factory=dict)
    solver: Dict[str, str] = field(default_factory=dict)
    mismatch_ranges: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main config
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """Complete benchmark configuration for one modality.

    Loaded from ``benchmarks/configs/<modality_id>.yaml``.
    """

    # Identity
    modality_id: str = ""
    display_name: str = ""
    category: str = ""
    canonical_dag: str = ""
    carrier: str = "Photon"
    maturity: str = "M0"
    forward_model_type: str = "linear_operator"
    default_solver: str = ""

    # Dimensions
    x_shape: List[int] = field(default_factory=lambda: [64, 64])
    y_shape: List[int] = field(default_factory=lambda: [64, 64])

    # Operator
    operator_id: str = ""
    has_dedicated_operator: bool = False
    theta: Dict[str, Any] = field(default_factory=dict)
    assets: Dict[str, Any] = field(default_factory=dict)
    graph_template_id: str = ""

    # Category module for shared physics
    category_module: str = ""

    # Data source
    data_source: DataSourceSpec = field(default_factory=DataSourceSpec)

    # Mismatch parameters
    mismatch_params: List[MismatchParam] = field(default_factory=list)

    # Solvers (keyed by tier)
    solvers: Dict[str, SolverSpec] = field(default_factory=dict)

    # Metrics
    metric_names: List[str] = field(default_factory=lambda: ["psnr", "ssim"])
    primary_metric: str = "psnr"
    acceptance_thresholds: Dict[str, float] = field(default_factory=dict)

    # Performance references
    reference_psnr: Optional[float] = None
    expected_psnr_range: Optional[Tuple[float, float]] = None

    # Source attribution
    source_attribution: SourceAttributionSpec = field(default_factory=SourceAttributionSpec)

    # Tier classification (A/B/C)
    tier: str = "C"

    @property
    def dims(self) -> Tuple[int, ...]:
        return tuple(self.x_shape)

    @classmethod
    def from_dict(cls, d: Dict) -> "BenchmarkConfig":
        """Build from a parsed YAML dict."""
        cfg = cls()

        # Identity
        cfg.modality_id = d.get("modality_id", "")
        cfg.display_name = d.get("display_name", cfg.modality_id)
        cfg.category = d.get("category", "")
        cfg.canonical_dag = d.get("canonical_dag", "")
        cfg.carrier = d.get("carrier", "Photon")
        cfg.maturity = d.get("maturity", "M0")
        cfg.forward_model_type = d.get("forward_model_type", "linear_operator")
        cfg.default_solver = d.get("default_solver", "")

        # Dimensions
        cfg.x_shape = d.get("x_shape", [64, 64])
        cfg.y_shape = d.get("y_shape", [64, 64])

        # Operator
        cfg.operator_id = d.get("operator_id", cfg.modality_id)
        cfg.has_dedicated_operator = d.get("has_dedicated_operator", False)
        cfg.theta = d.get("theta", {})
        cfg.assets = d.get("assets", {})
        cfg.graph_template_id = d.get("graph_template_id", "")

        # Category module
        cfg.category_module = d.get("category_module", "")

        # Data source
        ds = d.get("data_source", {})
        cfg.data_source = DataSourceSpec(
            priority=ds.get("priority", ["web", "experimental", "synthetic_web", "generated"]),
            dataset_id=ds.get("dataset_id", ""),
            dataset_url=ds.get("dataset_url", ""),
            fallback=ds.get("fallback", "generated"),
            synthetic_generator=ds.get("synthetic_generator", ""),
            citation=ds.get("citation", ""),
            license=ds.get("license", ""),
        )

        # Mismatch
        for mp in d.get("mismatch_params", []):
            cfg.mismatch_params.append(MismatchParam.from_dict(mp))

        # Solvers
        solvers_dict = d.get("solvers", {})
        for tier_name, solver_info in solvers_dict.items():
            if isinstance(solver_info, dict):
                cfg.solvers[tier_name] = SolverSpec.from_dict(solver_info, tier_name)

        # Metrics
        metrics_section = d.get("metrics", {})
        cfg.metric_names = metrics_section.get("names", ["psnr", "ssim"])
        cfg.primary_metric = metrics_section.get("primary", "psnr")
        cfg.acceptance_thresholds = metrics_section.get("thresholds", {})

        # Performance references
        cfg.reference_psnr = d.get("reference_psnr")
        psnr_range = d.get("expected_psnr_range")
        if psnr_range and len(psnr_range) == 2:
            cfg.expected_psnr_range = (float(psnr_range[0]), float(psnr_range[1]))

        # Source attribution
        sa = d.get("source_attribution", {})
        cfg.source_attribution = SourceAttributionSpec(
            ground_truth=sa.get("ground_truth", {}),
            forward_model=sa.get("forward_model", {}),
            solver=sa.get("solver", {}),
            mismatch_ranges=sa.get("mismatch_ranges", {}),
        )

        # Tier
        cfg.tier = d.get("tier", "C")

        return cfg


# ---------------------------------------------------------------------------
# Loader functions
# ---------------------------------------------------------------------------

def load_config(path: Path) -> BenchmarkConfig:
    """Load a single benchmark config from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return BenchmarkConfig.from_dict(raw)


def load_all_configs(config_dir: Optional[Path] = None) -> Dict[str, BenchmarkConfig]:
    """Load all YAML configs from the configs directory.

    Returns:
        Dict mapping modality_id to BenchmarkConfig.
    """
    if config_dir is None:
        config_dir = Path(__file__).parent.parent / "configs"
    configs: Dict[str, BenchmarkConfig] = {}
    for path in sorted(config_dir.glob("*.yaml")):
        if path.name.startswith("_"):
            continue
        cfg = load_config(path)
        configs[cfg.modality_id] = cfg
    return configs
