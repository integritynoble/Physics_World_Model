"""ExpandedConfig – loader for multi-variant benchmark configs.

Reads expanded YAML configs (e.g., cassi_expanded.yaml) that enumerate
ALL system variants, image sizes, compression ratios, noise levels,
and mismatch levels for a single modality. Generates the full
combinatorial test matrix.

Each test instance is a CaseInstance with a unique case_id.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

EXPANDED_CONFIG_DIR = Path(__file__).parent.parent / "expanded_configs"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Primitive:
    """One of the 11 typed primitives describing a physical element."""
    symbol: str   # Src, P, M, R, Pi, F, C, W, Sigma, S, D
    type: str     # e.g., binary_random_mask, prism, CCD
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemVariant:
    """One architecture variant of a modality (e.g., SD-CASSI, DD-CASSI)."""
    id: str
    name: str
    dag: str
    primitives: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    optical_elements: List[str] = field(default_factory=list)
    reference: str = ""


@dataclass
class ImageSize:
    """One image size configuration."""
    id: str
    x_shape: List[int]
    y_shape: Optional[List[int]] = None
    label: str = ""


@dataclass
class CompressionRatio:
    """One compression ratio configuration."""
    id: str
    shots: int = 1
    ratio: str = ""
    effective: float = 1.0


@dataclass
class NoiseLevel:
    """One noise level configuration."""
    id: str
    label: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MismatchLevel:
    """One mismatch severity level (M0-M4)."""
    id: str
    description: str = ""
    n_params_perturbed: Any = 0  # int or "all"


@dataclass
class MismatchParam:
    """One mismatch parameter with range."""
    name: str
    nominal: float
    range: Tuple[float, float]
    unit: str = ""
    primitive: str = ""
    description: str = ""


@dataclass
class DataSourceEntry:
    """One data source for the benchmark."""
    id: str
    type: str       # web, experimental, synthetic_web, generated
    label: str      # WEB, EXP, SYN-WEB, GEN
    url: str = ""
    description: str = ""
    citation: str = ""
    license: str = ""
    applies_to: Any = None  # list of variant IDs or "all_variants"


@dataclass
class CaseInstance:
    """A single test case in the benchmark matrix."""
    case_id: str
    modality_id: str
    benchmark: str       # B1, B2, B3, B4
    variant: SystemVariant
    image_size: ImageSize
    compression_ratio: Optional[CompressionRatio]
    noise_level: NoiseLevel
    mismatch_level: Optional[MismatchLevel]
    data_source: Optional[DataSourceEntry]

    # B1-specific
    prompt_difficulty: str = ""
    round_number: int = 1

    # B3/B4-specific: true-spec has exact param values
    true_spec_params: Dict[str, float] = field(default_factory=dict)

    def __repr__(self):
        return (f"CaseInstance({self.case_id}, {self.benchmark}, "
                f"variant={self.variant.id}, size={self.image_size.id})")


@dataclass
class ExpandedBenchmarkConfig:
    """Complete expanded config for one modality."""
    modality_id: str
    display_name: str
    category: str
    carrier: str
    maturity: str

    variants: Dict[str, SystemVariant] = field(default_factory=dict)
    image_sizes: Dict[str, ImageSize] = field(default_factory=dict)
    compression_ratios: Dict[str, CompressionRatio] = field(default_factory=dict)
    noise_levels: Dict[str, NoiseLevel] = field(default_factory=dict)
    mismatch_params: List[MismatchParam] = field(default_factory=list)
    mismatch_levels: Dict[str, MismatchLevel] = field(default_factory=dict)
    data_sources: List[DataSourceEntry] = field(default_factory=list)

    # Benchmark-specific config
    b1_config: Dict[str, Any] = field(default_factory=dict)
    b2_config: Dict[str, Any] = field(default_factory=dict)
    b3_config: Dict[str, Any] = field(default_factory=dict)
    b4_config: Dict[str, Any] = field(default_factory=dict)

    # Totals
    total_cases: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "ExpandedBenchmarkConfig":
        """Load an expanded config from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        config = cls(
            modality_id=raw.get("modality_id", ""),
            display_name=raw.get("display_name", ""),
            category=raw.get("category", ""),
            carrier=raw.get("carrier", ""),
            maturity=raw.get("maturity", "M0"),
        )

        # Parse variants
        for vid, vdata in raw.get("variants", {}).items():
            config.variants[vid] = SystemVariant(
                id=vdata.get("id", vid),
                name=vdata.get("name", vid),
                dag=vdata.get("dag", ""),
                primitives=vdata.get("primitives", {}),
                optical_elements=vdata.get("optical_elements", []),
                reference=vdata.get("reference", ""),
            )

        # Parse image sizes
        for sid, sdata in raw.get("image_sizes", {}).items():
            if isinstance(sdata, dict):
                x_shape = sdata.get("x_shape", sdata.get("recon", [256, 256]))
                y_shape = sdata.get("y_shape", sdata.get("sinogram"))
                config.image_sizes[sid] = ImageSize(
                    id=sid, x_shape=x_shape, y_shape=y_shape,
                    label=sdata.get("label", sid),
                )

        # Parse compression ratios
        for cid, cdata in raw.get("compression_ratios", {}).items():
            if isinstance(cdata, dict):
                config.compression_ratios[cid] = CompressionRatio(
                    id=cid,
                    shots=cdata.get("shots", 1),
                    ratio=str(cdata.get("ratio", "")),
                    effective=float(cdata.get("effective", 1.0)),
                )

        # Parse noise levels (handle both "noise_levels" and "dose_levels")
        noise_key = "noise_levels" if "noise_levels" in raw else "dose_levels"
        for nid, ndata in raw.get(noise_key, {}).items():
            if isinstance(ndata, dict):
                label = ndata.pop("label", nid)
                config.noise_levels[nid] = NoiseLevel(
                    id=nid, label=label, params=ndata,
                )

        # Parse mismatch params
        for mp in raw.get("mismatch_params", []):
            rng = mp.get("range", [0, 0])
            config.mismatch_params.append(MismatchParam(
                name=mp["name"],
                nominal=float(mp.get("nominal", 0)),
                range=(float(rng[0]), float(rng[1])),
                unit=mp.get("unit", ""),
                primitive=mp.get("primitive", ""),
                description=mp.get("description", ""),
            ))

        # Parse mismatch levels
        for mid, mdata in raw.get("mismatch_levels", {}).items():
            if isinstance(mdata, dict):
                config.mismatch_levels[mid] = MismatchLevel(
                    id=mid,
                    description=mdata.get("description", ""),
                    n_params_perturbed=mdata.get("n_params_perturbed", 0),
                )

        # Parse data sources
        for ds in raw.get("data_sources", []):
            config.data_sources.append(DataSourceEntry(
                id=ds.get("id", ""),
                type=ds.get("type", "generated"),
                label=ds.get("label", "GEN"),
                url=ds.get("url", ""),
                description=ds.get("description", ""),
                citation=ds.get("citation", ""),
                license=ds.get("license", ""),
                applies_to=ds.get("applies_to"),
            ))

        # Benchmark configs
        config.b1_config = raw.get("b1_design", {})
        config.b2_config = raw.get("b2_forward_reconstruct", {})
        config.b3_config = raw.get("b3_system_identification", {})
        config.b4_config = raw.get("b4_correct_diagnose", {})
        config.total_cases = raw.get("total_cases", {})

        return config

    def _find_data_source(self, variant_id: str) -> Optional[DataSourceEntry]:
        """Find the best data source for a given variant."""
        # Priority: WEB > EXP > SYN-WEB > GEN
        priority_order = ["WEB", "EXP", "SYN-WEB", "GEN"]
        for label in priority_order:
            for ds in self.data_sources:
                if ds.label != label:
                    continue
                applies = ds.applies_to
                if applies is None or applies == "all_variants":
                    return ds
                if isinstance(applies, list) and variant_id in applies:
                    return ds
        return self.data_sources[0] if self.data_sources else None

    def generate_b1_cases(self) -> List[CaseInstance]:
        """Generate all B1 (Design) test cases."""
        cases = []
        difficulties = ["easy", "medium", "hard", "adversarial"]
        n_rounds = self.b1_config.get("multi_round", 3)

        # Use first image size and noise level as defaults for B1
        default_size = list(self.image_sizes.values())[0] if self.image_sizes else ImageSize(id="default", x_shape=[256, 256])
        default_noise = list(self.noise_levels.values())[0] if self.noise_levels else NoiseLevel(id="clean")

        for variant in self.variants.values():
            for difficulty in difficulties:
                for round_num in range(1, n_rounds + 1):
                    case_id = f"{self.modality_id}_B1_{variant.id}_{difficulty}_r{round_num}"
                    cases.append(CaseInstance(
                        case_id=case_id,
                        modality_id=self.modality_id,
                        benchmark="B1",
                        variant=variant,
                        image_size=default_size,
                        compression_ratio=None,
                        noise_level=default_noise,
                        mismatch_level=None,
                        data_source=self._find_data_source(variant.id),
                        prompt_difficulty=difficulty,
                        round_number=round_num,
                    ))
        return cases

    def generate_b2_cases(self) -> List[CaseInstance]:
        """Generate all B2 (Forward + Reconstruct) test cases."""
        cases = []
        variants = list(self.variants.values())
        sizes = list(self.image_sizes.values())
        ratios = list(self.compression_ratios.values()) or [None]
        noises = list(self.noise_levels.values())
        mismatches = list(self.mismatch_levels.values())

        for variant, size, ratio, noise, mismatch in itertools.product(
            variants, sizes, ratios, noises, mismatches
        ):
            ratio_id = ratio.id if ratio else "default"
            case_id = (f"{self.modality_id}_B2_{variant.id}_{size.id}_"
                       f"{ratio_id}_{noise.id}_{mismatch.id}")
            cases.append(CaseInstance(
                case_id=case_id,
                modality_id=self.modality_id,
                benchmark="B2",
                variant=variant,
                image_size=size,
                compression_ratio=ratio,
                noise_level=noise,
                mismatch_level=mismatch,
                data_source=self._find_data_source(variant.id),
            ))
        return cases

    def generate_b3_cases(self) -> List[CaseInstance]:
        """Generate all B3 (System Identification) test cases.

        Same combinatorial space as B2 but each case has a true-spec
        with exact parameter values.
        """
        cases = []
        rng = np.random.default_rng(42)

        for b2_case in self.generate_b2_cases():
            case_id = b2_case.case_id.replace("_B2_", "_B3_")
            # Generate true-spec: sample exact values within ranges
            true_params = {}
            for mp in self.mismatch_params:
                true_params[mp.name] = float(
                    rng.uniform(mp.range[0], mp.range[1])
                )
            cases.append(CaseInstance(
                case_id=case_id,
                modality_id=self.modality_id,
                benchmark="B3",
                variant=b2_case.variant,
                image_size=b2_case.image_size,
                compression_ratio=b2_case.compression_ratio,
                noise_level=b2_case.noise_level,
                mismatch_level=b2_case.mismatch_level,
                data_source=b2_case.data_source,
                true_spec_params=true_params,
            ))
        return cases

    def generate_b4_cases(self) -> List[CaseInstance]:
        """Generate all B4 (Correct + Diagnose) test cases.

        Same as B3 plus correction targets.
        """
        cases = []
        for b3_case in self.generate_b3_cases():
            case_id = b3_case.case_id.replace("_B3_", "_B4_")
            cases.append(CaseInstance(
                case_id=case_id,
                modality_id=self.modality_id,
                benchmark="B4",
                variant=b3_case.variant,
                image_size=b3_case.image_size,
                compression_ratio=b3_case.compression_ratio,
                noise_level=b3_case.noise_level,
                mismatch_level=b3_case.mismatch_level,
                data_source=b3_case.data_source,
                true_spec_params=b3_case.true_spec_params,
            ))
        return cases

    def generate_all_cases(self) -> Dict[str, List[CaseInstance]]:
        """Generate all cases for all 4 benchmarks."""
        return {
            "B1": self.generate_b1_cases(),
            "B2": self.generate_b2_cases(),
            "B3": self.generate_b3_cases(),
            "B4": self.generate_b4_cases(),
        }

    def summary(self) -> Dict[str, Any]:
        """Return summary statistics."""
        all_cases = self.generate_all_cases()
        return {
            "modality_id": self.modality_id,
            "display_name": self.display_name,
            "category": self.category,
            "n_variants": len(self.variants),
            "n_sizes": len(self.image_sizes),
            "n_compression_ratios": len(self.compression_ratios),
            "n_noise_levels": len(self.noise_levels),
            "n_mismatch_levels": len(self.mismatch_levels),
            "n_mismatch_params": len(self.mismatch_params),
            "n_data_sources": len(self.data_sources),
            "cases": {k: len(v) for k, v in all_cases.items()},
            "total": sum(len(v) for v in all_cases.values()),
            "data_source_labels": sorted({ds.label for ds in self.data_sources}),
        }


# numpy needed for true-spec generation
try:
    import numpy as np
except ImportError:
    import random

    class _FakeRNG:
        def __init__(self, seed):
            random.seed(seed)
        def uniform(self, lo, hi):
            return random.uniform(lo, hi)

    class _FakeNP:
        class random:
            @staticmethod
            def default_rng(seed):
                return _FakeRNG(seed)

    np = _FakeNP()


def load_expanded_config(modality_id: str) -> ExpandedBenchmarkConfig:
    """Load an expanded config by modality ID."""
    path = EXPANDED_CONFIG_DIR / f"{modality_id}_expanded.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Expanded config not found: {path}. "
            f"Available: {[p.stem for p in EXPANDED_CONFIG_DIR.glob('*_expanded.yaml')]}"
        )
    return ExpandedBenchmarkConfig.from_yaml(path)


def load_all_expanded_configs() -> Dict[str, ExpandedBenchmarkConfig]:
    """Load all expanded configs from the expanded_configs directory."""
    configs = {}
    for path in sorted(EXPANDED_CONFIG_DIR.glob("*_expanded.yaml")):
        try:
            cfg = ExpandedBenchmarkConfig.from_yaml(path)
            configs[cfg.modality_id] = cfg
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
    return configs
