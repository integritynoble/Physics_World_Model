"""DataSource – multi-strategy data acquisition for benchmarks.

Follows the sourcing priority:
  1. web        – download from known dataset URLs
  2. experimental – load from local datasets/ directory
  3. synthetic_web – download synthetic from repositories
  4. generated  – create via category-module synthetic generator
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from benchmarks.framework.source_attribution import SourceAttribution, SourceRef, SourceType


# ---------------------------------------------------------------------------
# DataResult – what acquire() returns
# ---------------------------------------------------------------------------

@dataclass
class DataResult:
    """Result of data acquisition."""
    x_true: np.ndarray             # Ground truth signal
    source_type: SourceType        # How it was obtained
    reference: str = ""            # Citation / URL
    metadata: Dict[str, Any] = None  # Extra info (dataset name, etc.)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ---------------------------------------------------------------------------
# DataSource
# ---------------------------------------------------------------------------

class DataSource:
    """Multi-strategy data acquisition following sourcing priority."""

    # Class-level cache dir
    _cache_root: Path = Path(__file__).parent.parent / "results" / ".data_cache"

    def __init__(
        self,
        priority: List[str],
        dataset_id: str = "",
        dataset_url: str = "",
        fallback: str = "generated",
        citation: str = "",
        license_: str = "",
        synthetic_generator: str = "",
        category_module: str = "",
        dims: Tuple[int, ...] = (64, 64),
    ):
        self.priority = priority
        self.dataset_id = dataset_id
        self.dataset_url = dataset_url
        self.fallback = fallback
        self.citation = citation
        self.license_ = license_
        self.synthetic_generator = synthetic_generator
        self.category_module = category_module
        self.dims = dims

    def acquire(self) -> DataResult:
        """Try each source strategy in priority order.

        Returns:
            DataResult with ground truth array and provenance.
        """
        strategies = {
            "web": self._try_web,
            "experimental": self._try_experimental,
            "synthetic_web": self._try_synthetic_web,
            "generated": self._try_generated,
        }
        for source_name in self.priority:
            fn = strategies.get(source_name)
            if fn is None:
                continue
            result = fn()
            if result is not None:
                return result

        # Ultimate fallback: generate a simple test pattern
        return self._generate_fallback()

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _try_web(self) -> Optional[DataResult]:
        """Try to load from a web dataset URL."""
        if not self.dataset_url:
            return None
        cache_path = self._cache_path(self.dataset_url)
        if cache_path.exists():
            try:
                arr = np.load(cache_path)
                return DataResult(
                    x_true=arr,
                    source_type=SourceType.web,
                    reference=self.citation or self.dataset_url,
                    metadata={"dataset_id": self.dataset_id, "cached": True},
                )
            except Exception:
                pass
        # Attempt download
        try:
            self._cache_root.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(self.dataset_url, str(cache_path))
            arr = np.load(cache_path)
            return DataResult(
                x_true=arr,
                source_type=SourceType.web,
                reference=self.citation or self.dataset_url,
                metadata={"dataset_id": self.dataset_id},
            )
        except Exception:
            return None

    def _try_experimental(self) -> Optional[DataResult]:
        """Try to load real experimental data from local datasets/ directory."""
        if not self.dataset_id:
            return None
        datasets_dir = Path(__file__).parent.parent.parent / "datasets"
        if not datasets_dir.exists():
            return None
        # Look for modality-specific directory or .npy file
        for pattern in [
            self.dataset_id,
            self.dataset_id + ".npy",
            f"*{self.dataset_id}*",
        ]:
            matches = list(datasets_dir.glob(pattern))
            if matches:
                p = matches[0]
                if p.is_file() and p.suffix == ".npy":
                    arr = np.load(p)
                    return DataResult(
                        x_true=arr,
                        source_type=SourceType.experimental,
                        reference=f"local:{p.name}",
                        metadata={"path": str(p)},
                    )
                if p.is_dir():
                    # Look for first .npy in directory
                    npys = list(p.glob("*.npy"))
                    if npys:
                        arr = np.load(npys[0])
                        return DataResult(
                            x_true=arr,
                            source_type=SourceType.experimental,
                            reference=f"local:{p.name}/{npys[0].name}",
                            metadata={"path": str(npys[0])},
                        )
        return None

    def _try_synthetic_web(self) -> Optional[DataResult]:
        """Try to download synthetic data from known repositories."""
        # Synthetic web sources are just alternate URLs; same mechanism as web
        # but flagged differently for provenance tracking
        if not self.dataset_url:
            return None
        result = self._try_web()
        if result is not None:
            result.source_type = SourceType.synthetic_web
        return result

    def _try_generated(self) -> Optional[DataResult]:
        """Generate synthetic data via category module."""
        if self.category_module:
            try:
                gen = _load_category_generator(self.category_module)
                if gen is not None:
                    arr = gen(self.dims)
                    return DataResult(
                        x_true=arr,
                        source_type=SourceType.generated,
                        reference=f"category:{self.category_module}",
                        metadata={"generator": self.category_module},
                    )
            except Exception:
                pass
        return self._generate_fallback()

    def _generate_fallback(self) -> DataResult:
        """Last resort: Shepp-Logan phantom or test pattern."""
        arr = _generate_default_phantom(self.dims)
        return DataResult(
            x_true=arr,
            source_type=SourceType.generated,
            reference="default_phantom",
            metadata={"generator": "default_phantom"},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cache_path(self, url: str) -> Path:
        """Deterministic cache path from URL."""
        h = hashlib.sha256(url.encode()).hexdigest()[:16]
        return self._cache_root / f"{h}.npy"


# ---------------------------------------------------------------------------
# Default phantom generator
# ---------------------------------------------------------------------------

def _generate_default_phantom(dims: Tuple[int, ...]) -> np.ndarray:
    """Generate a default Shepp-Logan-like phantom.

    Works for 2-D ``(H, W)`` or 3-D ``(H, W, C)`` shapes.
    """
    H = dims[0] if len(dims) >= 1 else 64
    W = dims[1] if len(dims) >= 2 else H
    x = np.zeros((H, W), dtype=np.float32)

    # Ellipse parameters: (cx, cy, rx, ry, intensity)
    ellipses = [
        (0.0, 0.0, 0.69, 0.92, 1.0),
        (0.0, -0.0184, 0.6624, 0.874, -0.8),
        (0.22, 0.0, 0.11, 0.31, -0.2),
        (-0.22, 0.0, 0.16, 0.41, -0.2),
        (0.0, 0.35, 0.21, 0.25, 0.1),
        (0.0, 0.1, 0.046, 0.046, 0.1),
        (0.0, -0.1, 0.046, 0.046, 0.1),
        (-0.08, -0.605, 0.046, 0.023, 0.1),
        (0.0, -0.605, 0.023, 0.023, 0.1),
        (0.06, -0.605, 0.023, 0.046, 0.1),
    ]

    yy = np.linspace(-1, 1, H)
    xx = np.linspace(-1, 1, W)
    X, Y = np.meshgrid(xx, yy)

    for cx, cy, rx, ry, intensity in ellipses:
        mask = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 <= 1.0
        x[mask] += intensity

    x = np.clip(x, 0, 1).astype(np.float32)

    # For 3-D, replicate with slight variation per channel
    if len(dims) >= 3:
        C = dims[2]
        x3d = np.zeros((H, W, C), dtype=np.float32)
        for c in range(C):
            scale = 0.8 + 0.4 * c / max(C - 1, 1)
            x3d[..., c] = np.clip(x * scale, 0, 1)
        return x3d

    return x


def _load_category_generator(module_name: str):
    """Dynamically load a category module's generate_phantom function."""
    try:
        import importlib
        mod = importlib.import_module(f"benchmarks.categories.{module_name}")
        return getattr(mod, "generate_phantom", None)
    except (ImportError, AttributeError):
        return None
