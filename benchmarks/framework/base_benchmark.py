"""BaseBenchmark – template-driven runner for all 168 modalities.

Orchestrates data acquisition, operator construction, forward/reconstruct,
mismatch injection, grid-search correction, and result reporting.

Each modality is driven entirely by its YAML config — no per-modality
subclass is required (though category modules provide shared physics).
"""

from __future__ import annotations

import importlib
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from benchmarks.framework.benchmark_config import BenchmarkConfig, MismatchParam
from benchmarks.framework.data_source import DataSource, DataResult
from benchmarks.framework.metrics import MetricSet, compute_psnr, compute_ssim
from benchmarks.framework.mismatch_engine import (
    MismatchEngine,
    MismatchResult,
    MismatchScenario,
    GridSearchResult,
)
from benchmarks.framework.report_writer import ReportWriter, RunBundle
from benchmarks.framework.source_attribution import (
    SourceAttribution,
    SourceRef,
    SourceType,
)

logger = logging.getLogger(__name__)

# Ensure pwm_core is importable
_PWM_ROOT = Path(__file__).parent.parent.parent / "packages" / "pwm_core"
if str(_PWM_ROOT) not in sys.path:
    sys.path.insert(0, str(_PWM_ROOT))


def _import_build_benchmark_operator():
    """Import build_benchmark_operator handling module path conflicts.

    The top-level ``benchmarks/`` package can shadow
    ``packages/pwm_core/benchmarks/``, so we use importlib with an
    explicit file path.
    """
    import importlib.util
    helpers_path = _PWM_ROOT / "benchmarks" / "benchmark_helpers.py"
    if helpers_path.exists():
        spec = importlib.util.spec_from_file_location(
            "pwm_benchmark_helpers", str(helpers_path),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.build_benchmark_operator
    raise ImportError(f"Cannot find benchmark_helpers.py at {helpers_path}")


# ---------------------------------------------------------------------------
# BaseBenchmark
# ---------------------------------------------------------------------------

class BaseBenchmark:
    """Config-driven benchmark runner for a single modality.

    Lifecycle::

        bench = BaseBenchmark(config)
        result = bench.run(level="M0")

    Override hooks for custom behaviour:

    * ``_build_operator(theta)`` — custom operator construction
    * ``_forward(operator, x)``  — custom forward model
    * ``_reconstruct(operator, y)`` — custom reconstruction
    * ``_generate_phantom(dims)`` — custom phantom generation
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        output_dir: Optional[Path] = None,
        verbose: bool = True,
        seed: int = 42,
    ):
        self.cfg = config
        self.verbose = verbose
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.writer = ReportWriter(output_dir)

        # Build metric set from config
        self.metric_set = MetricSet(
            names=config.metric_names,
            primary=config.primary_metric,
            thresholds=config.acceptance_thresholds,
        )

        # Mismatch engine
        self.mismatch_engine = MismatchEngine(config.mismatch_params, self.metric_set)

        # Cache
        self._operator = None
        self._x_true = None
        self._data_result: Optional[DataResult] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, level: str = "M0", dry_run: bool = False) -> RunBundle:
        """Run the benchmark at the specified maturity level.

        Args:
            level: ``"M0"`` through ``"M4"``.
            dry_run: If True, validate config loading and operator build
                     without running reconstruction.

        Returns:
            ``RunBundle`` with all results and source attribution.
        """
        t0 = time.time()
        self.log(f"[{self.cfg.modality_id}] Starting {level} benchmark (tier {self.cfg.tier})")

        bundle = RunBundle(
            modality_id=self.cfg.modality_id,
            level=level,
            solver=self.cfg.default_solver,
            tier=self.cfg.tier,
            reference_psnr=self.cfg.reference_psnr,
            expected_psnr_range=(
                list(self.cfg.expected_psnr_range)
                if self.cfg.expected_psnr_range else None
            ),
            config_path=str(self.cfg.modality_id),
        )

        # 1. Acquire data
        self.log(f"  Acquiring data...")
        data_result = self.acquire_data()
        self._data_result = data_result
        self._x_true = data_result.x_true
        self.log(f"  Data source: {data_result.source_type.value} "
                 f"({data_result.reference}), shape={data_result.x_true.shape}")

        if dry_run:
            # Validate operator builds
            self.log(f"  [dry-run] Building operator...")
            op = self.build_operator()
            self.log(f"  [dry-run] Operator built: {type(op).__name__}")
            bundle.wall_time_s = time.time() - t0
            return bundle

        # 2. Build operator and run B2 (forward + reconstruct)
        self.log(f"  Running B2: forward + reconstruct...")
        b2_metrics, per_algo = self.run_b2(data_result.x_true)
        bundle.metrics = b2_metrics
        bundle.per_algorithm = per_algo

        # 3. Run mismatch scenarios if level >= M1
        if level in ("M1", "M2", "M3", "M4") and self.cfg.mismatch_params:
            self.log(f"  Running mismatch scenarios ({level})...")
            mismatch_results = self.run_mismatch_scenarios(
                data_result.x_true, level,
            )
            bundle.mismatch_results = [
                {
                    "name": mr.scenario.name,
                    "params": mr.scenario.params,
                    "psnr": mr.metrics.get("psnr"),
                    "ssim": mr.metrics.get("ssim"),
                    "psnr_drop": mr.psnr_drop,
                }
                for mr in mismatch_results
            ]

        # 4. Run B4 grid-search correction if level >= M2
        if level in ("M2", "M3", "M4") and self.cfg.mismatch_params:
            self.log(f"  Running B4: grid-search correction...")
            gs_result = self.run_b4(data_result.x_true, bundle.metrics.get("psnr", 0))
            bundle.grid_search_result = {
                "best_params": gs_result.best_params,
                "best_metric": gs_result.best_metric,
                "metric_name": gs_result.metric_name,
                "rho": gs_result.rho,
            }
            bundle.rho = gs_result.rho

        # 5. Source attribution
        bundle.source_attribution = self._build_source_attribution(data_result).to_dict()

        bundle.wall_time_s = time.time() - t0
        self.log(f"  Done in {bundle.wall_time_s:.1f}s. "
                 f"PSNR={bundle.metrics.get('psnr', 0):.2f} dB")

        # Save results
        json_path = self.writer.save_json(bundle)
        md_path = self.writer.save_markdown(bundle)
        self.log(f"  Saved: {json_path}")

        return bundle

    # ------------------------------------------------------------------
    # Step 1: Data acquisition
    # ------------------------------------------------------------------

    def acquire_data(self) -> DataResult:
        """Acquire ground truth data following sourcing priority."""
        ds = self.cfg.data_source
        source = DataSource(
            priority=ds.priority,
            dataset_id=ds.dataset_id,
            dataset_url=ds.dataset_url,
            fallback=ds.fallback,
            citation=ds.citation,
            license_=ds.license,
            synthetic_generator=ds.synthetic_generator,
            category_module=self.cfg.category_module,
            dims=self.cfg.dims,
        )
        return source.acquire()

    # ------------------------------------------------------------------
    # Step 2: Operator construction
    # ------------------------------------------------------------------

    def build_operator(self, theta: Optional[Dict[str, Any]] = None):
        """Build the forward operator.

        Uses ``build_benchmark_operator()`` from the existing infrastructure,
        which tries graph-first → legacy operator fallback.
        """
        build_benchmark_operator = _import_build_benchmark_operator()

        effective_theta = dict(self.cfg.theta)
        if theta:
            effective_theta.update(theta)

        operator = build_benchmark_operator(
            modality=self.cfg.operator_id or self.cfg.modality_id,
            dims=self.cfg.dims,
            theta=effective_theta if effective_theta else None,
            assets=self.cfg.assets if self.cfg.assets else None,
        )
        self._operator = operator
        return operator

    # ------------------------------------------------------------------
    # Step 3: B2 – Forward + Reconstruct
    # ------------------------------------------------------------------

    def run_b2(self, x_true: np.ndarray) -> Tuple[Dict[str, float], Dict[str, Dict]]:
        """Run forward model and reconstruction with all configured solvers.

        Returns:
            (primary_metrics, per_algorithm_dict)
        """
        operator = self.build_operator()
        per_algorithm: Dict[str, Dict] = {}
        primary_metrics: Dict[str, float] = {}

        # Forward pass
        y = self._forward(operator, x_true)

        # Run each solver tier
        solver_tiers = self.cfg.solvers or {"default": None}
        for tier_name, solver_spec in solver_tiers.items():
            try:
                x_hat = self._reconstruct(operator, y, solver_spec)
                metrics = self.metric_set.evaluate(x_true, x_hat)
                algo_name = solver_spec.id if solver_spec else tier_name
                per_algorithm[algo_name] = {
                    "psnr": metrics.get("psnr", 0),
                    "ssim": metrics.get("ssim", 0),
                    "tier": tier_name,
                    "params": solver_spec.params if solver_spec else "0",
                }
                self.log(f"    {algo_name}: PSNR={metrics.get('psnr', 0):.2f} dB, "
                         f"SSIM={metrics.get('ssim', 0):.4f}")
            except Exception as e:
                self.log(f"    {tier_name}: failed ({e})")
                per_algorithm[tier_name] = {
                    "psnr": 0.0, "ssim": 0.0, "tier": tier_name,
                    "params": "0", "error": str(e),
                }

        # Primary metrics = best solver
        if per_algorithm:
            best = max(per_algorithm.values(), key=lambda v: v.get("psnr", 0))
            primary_metrics = {
                "psnr": best.get("psnr", 0),
                "ssim": best.get("ssim", 0),
            }

        return primary_metrics, per_algorithm

    # ------------------------------------------------------------------
    # Step 4: Mismatch scenarios
    # ------------------------------------------------------------------

    def run_mismatch_scenarios(
        self,
        x_true: np.ndarray,
        level: str = "M1",
    ) -> List[MismatchResult]:
        """Generate and evaluate mismatch scenarios."""
        scenarios = self.mismatch_engine.generate_scenarios(level)

        def run_fn(params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
            operator = self.build_operator(theta=params)
            y = self._forward(operator, x_true)
            x_hat = self._reconstruct(operator, y)
            return x_hat, y

        return self.mismatch_engine.evaluate_scenarios(scenarios, run_fn, x_true)

    # ------------------------------------------------------------------
    # Step 5: B4 – Grid-search correction
    # ------------------------------------------------------------------

    def run_b4(
        self,
        x_true: np.ndarray,
        nominal_psnr: float = 0.0,
    ) -> GridSearchResult:
        """Run grid-search correction over mismatch parameters."""
        # First evaluate under a mid-range mismatch
        mismatch_params = {}
        for p in self.cfg.mismatch_params:
            mismatch_params[p.name] = (p.range[0] + p.range[1]) / 2.0
        operator = self.build_operator(theta=mismatch_params)
        y_mismatch = self._forward(operator, x_true)

        # Mismatched reconstruction (operator built with wrong params, data from wrong params)
        x_hat_mm = self._reconstruct(operator, y_mismatch)
        mm_psnr = compute_psnr(x_true, x_hat_mm)

        def run_fn(params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
            op = self.build_operator(theta=params)
            x_hat = self._reconstruct(op, y_mismatch)
            return x_hat, y_mismatch

        return self.mismatch_engine.grid_search(
            run_fn=run_fn,
            x_true=x_true,
            n_grid=5,
            nominal_metric=nominal_psnr,
            mismatched_metric=mm_psnr,
        )

    # ------------------------------------------------------------------
    # Forward / Reconstruct hooks (overridable)
    # ------------------------------------------------------------------

    def _forward(self, operator, x: np.ndarray) -> np.ndarray:
        """Run forward model.  Override for custom physics."""
        # Try category module first
        if self.cfg.category_module:
            fwd_fn = self._load_category_forward()
            if fwd_fn is not None:
                return fwd_fn(operator, x)

        # Default: use operator.forward()
        try:
            y = operator.forward(x)
            if y is not None:
                return np.asarray(y)
        except Exception:
            pass

        # Fallback: treat operator as linear (A @ x.ravel())
        try:
            if hasattr(operator, "A"):
                return (operator.A @ x.ravel()).reshape(self.cfg.y_shape)
        except Exception:
            pass

        # Last resort: identity forward (for dry-run / template validation)
        return x.copy()

    def _reconstruct(
        self,
        operator,
        y: np.ndarray,
        solver_spec=None,
    ) -> np.ndarray:
        """Run reconstruction solver.  Override for custom solvers."""
        # Try loading solver from registry
        if solver_spec and solver_spec.module and solver_spec.function:
            try:
                mod = importlib.import_module(solver_spec.module)
                fn = getattr(mod, solver_spec.function)
                result = fn(y.astype(np.float32), operator, solver_spec.cfg)
                if isinstance(result, tuple):
                    return np.asarray(result[0])
                return np.asarray(result)
            except Exception as e:
                self.log(f"    Solver {solver_spec.id} import failed: {e}")

        # Fallback: adjoint (pseudo-inverse) reconstruction
        try:
            x_hat = operator.adjoint(y)
            if x_hat is not None:
                return np.asarray(x_hat)
        except Exception:
            pass

        # Fallback: A^T @ y
        try:
            if hasattr(operator, "A"):
                return (operator.A.T @ y.ravel()).reshape(self.cfg.x_shape)
        except Exception:
            pass

        return y.copy()

    def _generate_phantom(self, dims: Tuple[int, ...]) -> np.ndarray:
        """Generate a test phantom. Override for modality-specific phantoms."""
        from benchmarks.framework.data_source import _generate_default_phantom
        return _generate_default_phantom(dims)

    # ------------------------------------------------------------------
    # Category module loading
    # ------------------------------------------------------------------

    def _load_category_forward(self) -> Optional[Callable]:
        """Load forward function from category module."""
        try:
            mod = importlib.import_module(
                f"benchmarks.categories.{self.cfg.category_module}"
            )
            return getattr(mod, "forward", None)
        except (ImportError, AttributeError):
            return None

    def _load_category_adjoint(self) -> Optional[Callable]:
        """Load adjoint function from category module."""
        try:
            mod = importlib.import_module(
                f"benchmarks.categories.{self.cfg.category_module}"
            )
            return getattr(mod, "adjoint", None)
        except (ImportError, AttributeError):
            return None

    # ------------------------------------------------------------------
    # Source attribution
    # ------------------------------------------------------------------

    def _build_source_attribution(self, data_result: DataResult) -> SourceAttribution:
        """Build source attribution from config and data result."""
        attr = SourceAttribution()

        # Ground truth source
        attr.ground_truth = SourceRef(
            type=data_result.source_type,
            reference=data_result.reference,
        )

        # Forward model source
        sa = self.cfg.source_attribution
        if sa.forward_model:
            attr.forward_model = SourceRef(
                type=SourceType(sa.forward_model.get("type", "registry")),
                reference=sa.forward_model.get("reference", ""),
            )
        else:
            attr.forward_model = SourceRef(
                type=SourceType.registry,
                reference=f"pwm graph_templates.yaml / {self.cfg.graph_template_id}",
            )

        # Solver source
        if sa.solver:
            attr.solver = SourceRef(
                type=SourceType(sa.solver.get("type", "registry")),
                reference=sa.solver.get("reference", ""),
            )
        else:
            attr.solver = SourceRef(
                type=SourceType.registry,
                reference=f"pwm solver_registry.yaml / {self.cfg.default_solver}",
            )

        # Mismatch ranges
        if sa.mismatch_ranges:
            attr.mismatch_ranges = SourceRef(
                type=SourceType(sa.mismatch_ranges.get("type", "registry")),
                reference=sa.mismatch_ranges.get("reference", ""),
            )
        else:
            attr.mismatch_ranges = SourceRef(
                type=SourceType.registry,
                reference="pwm modality_benchmarks spec",
            )

        return attr

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(self, msg: str):
        if self.verbose:
            print(msg)
