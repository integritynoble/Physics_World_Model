"""ExpandedBenchmarkRunner – execute CaseInstance objects from expanded configs.

Consumes ``ExpandedBenchmarkConfig`` and runs individual ``CaseInstance``
objects through B1–B4 benchmarks, reusing existing framework components
(``DataSource``, ``MetricSet``, ``MismatchEngine``, ``ReportWriter``).

Usage::

    from benchmarks.framework.expanded_config import load_expanded_config
    from benchmarks.framework.expanded_runner import ExpandedBenchmarkRunner

    ecfg = load_expanded_config("cassi")
    cases = ecfg.generate_b2_cases()[:5]

    runner = ExpandedBenchmarkRunner(ecfg)
    results = runner.run_cases(cases)
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from benchmarks.framework.benchmark_config import (
    BenchmarkConfig,
    MismatchParam as ConfigMismatchParam,
    load_config,
)
from benchmarks.framework.data_source import DataSource, DataResult
from benchmarks.framework.expanded_config import (
    CaseInstance,
    ExpandedBenchmarkConfig,
    MismatchParam as ExpandedMismatchParam,
)
from benchmarks.framework.expanded_result import CaseResult
from benchmarks.framework.metrics import MetricSet, compute_psnr
from benchmarks.framework.mismatch_engine import MismatchEngine, GridSearchResult

logger = logging.getLogger(__name__)

# Ensure pwm_core is importable
_PWM_ROOT = Path(__file__).parent.parent.parent / "packages" / "pwm_core"
if str(_PWM_ROOT) not in sys.path:
    sys.path.insert(0, str(_PWM_ROOT))


def _import_build_benchmark_operator():
    """Import build_benchmark_operator from pwm_core."""
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
# ExpandedBenchmarkRunner
# ---------------------------------------------------------------------------

class ExpandedBenchmarkRunner:
    """Runner that executes ``CaseInstance`` objects from expanded configs.

    Unlike ``BaseBenchmark`` (which consumes a single ``BenchmarkConfig``),
    this runner is parameterised per-case from ``CaseInstance`` fields and
    supports the full combinatorial matrix.

    Args:
        expanded_config: The expanded benchmark config for this modality.
        simple_config: Optional simple ``BenchmarkConfig`` as fallback for
            theta, solvers, etc.
        output_dir: Where to save per-case result JSON files.
        verbose: Print progress messages.
        seed: Random seed.
    """

    def __init__(
        self,
        expanded_config: ExpandedBenchmarkConfig,
        simple_config: Optional[BenchmarkConfig] = None,
        output_dir: Optional[Path] = None,
        verbose: bool = True,
        seed: int = 42,
    ):
        self.ecfg = expanded_config
        self.simple_cfg = simple_config
        self.output_dir = output_dir or Path(__file__).parent.parent / "results"
        self.verbose = verbose
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Cache for expensive objects
        self._data_cache: Dict[str, DataResult] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_case(self, case: CaseInstance, dry_run: bool = False) -> CaseResult:
        """Execute a single case and return the result.

        Args:
            case: The ``CaseInstance`` to run.
            dry_run: If True, validate config and operator build only.

        Returns:
            ``CaseResult`` with metrics and status.
        """
        t0 = time.time()
        result = CaseResult(
            case_id=case.case_id,
            modality_id=case.modality_id,
            benchmark=case.benchmark,
            variant_id=case.variant.id,
            image_size_id=case.image_size.id,
            noise_level_id=case.noise_level.id,
            mismatch_level_id=case.mismatch_level.id if case.mismatch_level else "M0",
        )

        try:
            if dry_run:
                self._dry_run_case(case, result)
            elif case.benchmark == "B1":
                self._run_b1(case, result)
            elif case.benchmark == "B2":
                self._run_b2(case, result)
            elif case.benchmark == "B3":
                self._run_b3(case, result)
            elif case.benchmark == "B4":
                self._run_b4(case, result)
            else:
                result.status = "error"
                result.error_message = f"Unknown benchmark: {case.benchmark}"
        except Exception as e:
            result.status = "error"
            result.error_message = str(e)
            logger.error(f"Case {case.case_id} failed: {e}")

        result.wall_time_s = time.time() - t0
        self._save_case_result(result)
        return result

    def run_cases(
        self,
        cases: List[CaseInstance],
        parallel: int = 1,
        dry_run: bool = False,
    ) -> List[CaseResult]:
        """Execute a batch of cases.

        Args:
            cases: Cases to run.
            parallel: Number of parallel workers (1 = sequential).
            dry_run: If True, validate only.

        Returns:
            List of ``CaseResult``.
        """
        if parallel > 1:
            return self._run_parallel(cases, parallel, dry_run)

        results = []
        for i, case in enumerate(cases, 1):
            self.log(f"[{i}/{len(cases)}] {case.case_id}")
            result = self.run_case(case, dry_run=dry_run)
            psnr_val = result.metrics.get("psnr")
            psnr_str = f" PSNR={psnr_val:.2f}" if psnr_val else ""
            self.log(f"  -> {result.status}{psnr_str} ({result.wall_time_s:.1f}s)")
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Dry run
    # ------------------------------------------------------------------

    def _dry_run_case(self, case: CaseInstance, result: CaseResult):
        """Validate config loading and operator build without reconstruction."""
        self.log(f"  [dry-run] Acquiring data for {case.variant.id}...")
        data = self._acquire_data(case)
        self.log(f"  [dry-run] Data shape: {data.x_true.shape}, source: {data.source_type.value}")

        self.log(f"  [dry-run] Building operator...")
        op = self._build_operator(case)
        self.log(f"  [dry-run] Operator type: {type(op).__name__}")

        result.status = "success"
        result.metrics = {"validated": 1.0}

    # ------------------------------------------------------------------
    # B1: Design (stub — needs LLM)
    # ------------------------------------------------------------------

    def _run_b1(self, case: CaseInstance, result: CaseResult):
        """B1 Design benchmark — requires LLM, so mark as skipped."""
        prompt_text = (
            f"Design a {case.variant.name} system with the following primitives: "
            f"{list(case.variant.primitives.keys())}. "
            f"Difficulty: {case.prompt_difficulty}, round {case.round_number}."
        )
        result.b1_scores = {
            "prompt": prompt_text,
            "difficulty": case.prompt_difficulty,
            "round": case.round_number,
            "variant": case.variant.id,
        }
        result.status = "skipped"
        result.error_message = "B1 requires LLM evaluation (not yet implemented)"
        self.log(f"  [B1] Skipped (LLM required). Prompt: {prompt_text[:80]}...")

    # ------------------------------------------------------------------
    # B2: Forward + Reconstruct
    # ------------------------------------------------------------------

    def _run_b2(self, case: CaseInstance, result: CaseResult):
        """B2: forward model, noise injection, optional mismatch, reconstruct, metrics."""
        # 1. Acquire data
        data = self._acquire_data(case)
        x_true = self._resize_data(data.x_true, case.image_size.x_shape)

        # 2. Build operator (nominal)
        operator = self._build_operator(case)

        # 3. Forward pass
        y = self._forward(operator, x_true, case)

        # 4. Noise injection
        y_noisy = self._apply_noise(y, case)

        # 5. Mismatch (if level > M0, perturb operator for reconstruction)
        recon_operator = operator
        mismatch_params_used = {}
        if case.mismatch_level and case.mismatch_level.id != "M0_nominal":
            mismatch_params_used = self._generate_mismatch_theta(case)
            if mismatch_params_used:
                recon_operator = self._build_operator(case, theta_overrides=mismatch_params_used)

        # 6. Reconstruct
        x_hat = self._reconstruct(recon_operator, y_noisy, case)

        # 7. Metrics (match shapes if reconstruction is lower-dimensional)
        x_true_m, x_hat_m = self._match_shapes(x_true, x_hat)
        metric_set = self._build_metric_set(case.benchmark)
        metrics = metric_set.evaluate(x_true_m, x_hat_m)
        result.metrics = metrics
        result.per_algorithm = {
            "default": {
                "psnr": metrics.get("psnr", 0),
                "ssim": metrics.get("ssim", 0),
                "variant": case.variant.id,
            },
        }
        if mismatch_params_used:
            result.mismatch_results = [{
                "level": case.mismatch_level.id if case.mismatch_level else "M0",
                "params": mismatch_params_used,
            }]
        result.status = "success"

    # ------------------------------------------------------------------
    # B3: System Identification
    # ------------------------------------------------------------------

    def _run_b3(self, case: CaseInstance, result: CaseResult):
        """B3: forward with true params, grid search to estimate params."""
        # 1. Acquire data and build operator with TRUE params
        data = self._acquire_data(case)
        x_true = self._resize_data(data.x_true, case.image_size.x_shape)

        true_theta = self._true_spec_to_theta(case)
        operator_true = self._build_operator(case, theta_overrides=true_theta)

        # 2. Forward with true params + noise
        y = self._forward(operator_true, x_true, case)
        y_noisy = self._apply_noise(y, case)

        # 3. Grid search over parameter space to identify params
        mismatch_params = self._expanded_to_config_mismatch_params()
        metric_set = self._build_metric_set(case.benchmark)
        engine = MismatchEngine(mismatch_params, metric_set)

        # Pilot reconstruct to determine output shape for metric matching
        x_hat_pilot = self._reconstruct(operator_true, y_noisy, case)
        x_true_matched, _ = self._match_shapes(x_true, x_hat_pilot)

        def run_fn(params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
            op = self._build_operator(case, theta_overrides=params)
            x_hat = self._reconstruct(op, y_noisy, case)
            _, x_hat = self._match_shapes(x_true, x_hat)
            return x_hat, y_noisy

        gs_result = engine.grid_search(
            run_fn=run_fn,
            x_true=x_true_matched,
            n_grid=5,
        )

        # 4. Compute estimation errors
        result.true_params = dict(case.true_spec_params)
        result.estimated_params = gs_result.best_params
        result.param_errors = {
            k: abs(gs_result.best_params.get(k, 0) - case.true_spec_params.get(k, 0))
            for k in case.true_spec_params
        }
        result.metrics = {"psnr": gs_result.best_metric}
        result.grid_search_result = {
            "best_params": gs_result.best_params,
            "best_metric": gs_result.best_metric,
            "metric_name": gs_result.metric_name,
            "grid_shape": list(gs_result.grid_shape),
        }
        result.status = "success"

    # ------------------------------------------------------------------
    # B4: Correct + Diagnose
    # ------------------------------------------------------------------

    def _run_b4(self, case: CaseInstance, result: CaseResult):
        """B4: forward with true params, reconstruct with nominal, grid search correction."""
        # 1. Acquire data
        data = self._acquire_data(case)
        x_true = self._resize_data(data.x_true, case.image_size.x_shape)

        # 2. Forward with true (unknown-to-solver) params
        true_theta = self._true_spec_to_theta(case)
        operator_true = self._build_operator(case, theta_overrides=true_theta)
        y = self._forward(operator_true, x_true, case)
        y_noisy = self._apply_noise(y, case)

        # 3. Reconstruct with nominal (mismatched) operator
        operator_nominal = self._build_operator(case)
        x_hat_nominal = self._reconstruct(operator_nominal, y_noisy, case)

        # 4. Metrics under nominal (mismatched) reconstruction
        #    Shape-match x_true to reconstruction output dims
        metric_set = self._build_metric_set(case.benchmark)
        x_true_matched, x_hat_nominal_m = self._match_shapes(x_true, x_hat_nominal)
        nominal_metrics = metric_set.evaluate(x_true_matched, x_hat_nominal_m)
        mismatched_psnr = nominal_metrics.get("psnr", 0.0)

        # 5. Reconstruct with true operator (oracle baseline)
        x_hat_oracle = self._reconstruct(operator_true, y_noisy, case)
        _, x_hat_oracle_m = self._match_shapes(x_true, x_hat_oracle)
        oracle_metrics = metric_set.evaluate(x_true_matched, x_hat_oracle_m)
        oracle_psnr = oracle_metrics.get("psnr", 0.0)

        # 6. Grid search correction
        mismatch_params = self._expanded_to_config_mismatch_params()
        engine = MismatchEngine(mismatch_params, metric_set)

        def run_fn(params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
            op = self._build_operator(case, theta_overrides=params)
            x_hat = self._reconstruct(op, y_noisy, case)
            _, x_hat = self._match_shapes(x_true, x_hat)
            return x_hat, y_noisy

        gs_result = engine.grid_search(
            run_fn=run_fn,
            x_true=x_true_matched,
            n_grid=5,
            nominal_metric=oracle_psnr,
            mismatched_metric=mismatched_psnr,
        )

        # 7. Populate result
        result.rho = gs_result.rho
        result.psnr_improvement = gs_result.best_metric - mismatched_psnr
        result.metrics = {
            "psnr_mismatched": mismatched_psnr,
            "psnr_corrected": gs_result.best_metric,
            "psnr_oracle": oracle_psnr,
            "rho": gs_result.rho,
        }
        result.grid_search_result = {
            "best_params": gs_result.best_params,
            "best_metric": gs_result.best_metric,
            "metric_name": gs_result.metric_name,
            "rho": gs_result.rho,
            "grid_shape": list(gs_result.grid_shape),
        }
        result.true_params = dict(case.true_spec_params)
        result.estimated_params = gs_result.best_params
        result.status = "success"

    # ------------------------------------------------------------------
    # Data acquisition
    # ------------------------------------------------------------------

    def _acquire_data(self, case: CaseInstance) -> DataResult:
        """Acquire ground-truth data, using cache when possible."""
        cache_key = f"{case.modality_id}_{case.variant.id}"
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]

        # Bridge CaseInstance.data_source -> DataSource constructor
        ds_entry = case.data_source
        priority = ["generated"]
        url = ""
        citation = ""
        license_ = ""
        dataset_id = ""
        category_module = ""

        if ds_entry:
            type_to_priority = {
                "web": ["web", "generated"],
                "experimental": ["experimental", "generated"],
                "synthetic_web": ["synthetic_web", "generated"],
                "generated": ["generated"],
            }
            priority = type_to_priority.get(ds_entry.type, ["generated"])
            url = ds_entry.url
            citation = ds_entry.citation
            license_ = ds_entry.license
            dataset_id = ds_entry.id

        # Try to find category_module from simple config
        if self.simple_cfg:
            category_module = self.simple_cfg.category_module
            if not dataset_id:
                dataset_id = self.simple_cfg.data_source.dataset_id
            if not url:
                url = self.simple_cfg.data_source.dataset_url

        dims = tuple(case.image_size.x_shape)
        source = DataSource(
            priority=priority,
            dataset_id=dataset_id,
            dataset_url=url,
            fallback="generated",
            citation=citation,
            license_=license_,
            category_module=category_module,
            dims=dims,
        )
        result = source.acquire()
        self._data_cache[cache_key] = result
        return result

    # ------------------------------------------------------------------
    # Operator construction
    # ------------------------------------------------------------------

    def _build_operator(
        self,
        case: CaseInstance,
        theta_overrides: Optional[Dict[str, Any]] = None,
    ):
        """Build forward operator from variant primitives + optional overrides.

        Flattens ``case.variant.primitives`` into a theta dict, merges with
        the simple config theta (if available), then applies overrides.
        """
        build_benchmark_operator = _import_build_benchmark_operator()

        # Start with simple config theta as base
        theta: Dict[str, Any] = {}
        if self.simple_cfg:
            theta.update(self.simple_cfg.theta)

        # Flatten variant primitives into theta
        # e.g., {Src: {type: halogen}} -> {Src_type: halogen}
        for symbol, prim_data in case.variant.primitives.items():
            if isinstance(prim_data, dict):
                for k, v in prim_data.items():
                    theta[f"{symbol}_{k}"] = v

        # Apply overrides (e.g., mismatch params or true-spec params)
        if theta_overrides:
            theta.update(theta_overrides)

        modality = self.simple_cfg.operator_id if self.simple_cfg else case.modality_id
        dims = tuple(case.image_size.x_shape)
        assets = self.simple_cfg.assets if self.simple_cfg else {}

        return build_benchmark_operator(
            modality=modality or case.modality_id,
            dims=dims,
            theta=theta if theta else None,
            assets=assets if assets else None,
        )

    # ------------------------------------------------------------------
    # Forward model
    # ------------------------------------------------------------------

    def _forward(
        self,
        operator,
        x: np.ndarray,
        case: CaseInstance,
    ) -> np.ndarray:
        """Run forward model with fallback chain."""
        # Try category module first
        cat_fwd = self._load_category_function("forward")
        if cat_fwd is not None:
            try:
                return cat_fwd(operator, x)
            except Exception:
                pass

        # operator.forward()
        try:
            y = operator.forward(x)
            if y is not None:
                return np.asarray(y)
        except Exception:
            pass

        # Linear: A @ x.ravel()
        try:
            if hasattr(operator, "A"):
                y_shape = case.image_size.y_shape or case.image_size.x_shape
                return (operator.A @ x.ravel()).reshape(y_shape)
        except Exception:
            pass

        # Identity fallback
        return x.copy()

    # ------------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------------

    def _reconstruct(
        self,
        operator,
        y: np.ndarray,
        case: CaseInstance,
        solver_spec=None,
    ) -> np.ndarray:
        """Run reconstruction with fallback chain."""
        # Try solver from simple config
        if solver_spec is None and self.simple_cfg and self.simple_cfg.solvers:
            # Use the first available solver
            for _, spec in self.simple_cfg.solvers.items():
                solver_spec = spec
                break

        if solver_spec and solver_spec.module and solver_spec.function:
            try:
                mod = importlib.import_module(solver_spec.module)
                fn = getattr(mod, solver_spec.function)
                result = fn(y.astype(np.float32), operator, solver_spec.cfg)
                if isinstance(result, tuple):
                    return np.asarray(result[0])
                return np.asarray(result)
            except Exception:
                pass

        # Category adjoint
        cat_adj = self._load_category_function("adjoint")
        if cat_adj is not None:
            try:
                return cat_adj(operator, y)
            except Exception:
                pass

        # operator.adjoint()
        try:
            x_hat = operator.adjoint(y)
            if x_hat is not None:
                return np.asarray(x_hat)
        except Exception:
            pass

        # A^T @ y
        try:
            if hasattr(operator, "A"):
                return (operator.A.T @ y.ravel()).reshape(case.image_size.x_shape)
        except Exception:
            pass

        return y.copy()

    # ------------------------------------------------------------------
    # Noise injection
    # ------------------------------------------------------------------

    def _apply_noise(self, y: np.ndarray, case: CaseInstance) -> np.ndarray:
        """Inject noise per case.noise_level.params."""
        params = case.noise_level.params
        if not params or case.noise_level.id == "clean":
            return y.copy()

        y_noisy = y.astype(np.float64).copy()

        # SNR-based additive Gaussian noise (universal)
        snr_db = params.get("snr_db")
        if snr_db is not None:
            signal_power = np.mean(y_noisy ** 2)
            if signal_power > 0:
                noise_power = signal_power / (10 ** (snr_db / 10))
                y_noisy += self.rng.randn(*y_noisy.shape) * np.sqrt(noise_power)

        # Shot noise (Poisson)
        photons = params.get("shot_noise_photons")
        if photons is not None and photons > 0:
            # Scale to photon counts, apply Poisson, scale back
            y_max = y_noisy.max()
            if y_max > 0:
                y_scaled = np.clip(y_noisy / y_max, 0, None) * photons
                y_noisy = self.rng.poisson(y_scaled).astype(np.float64) / photons * y_max

        # Read noise (Gaussian, in electron units)
        read_noise = params.get("read_noise_e")
        if read_noise is not None and read_noise > 0:
            y_noisy += self.rng.randn(*y_noisy.shape) * read_noise

        # CT-style photon count factor (dose reduction)
        pcf = params.get("photon_count_factor")
        if pcf is not None and pcf < 1.0:
            # Reduce effective photon count -> increase noise variance
            y_max = max(np.abs(y_noisy).max(), 1e-10)
            y_scaled = np.clip(y_noisy / y_max, 0, None)
            # Poisson noise at reduced dose
            counts = y_scaled * 1e4 * pcf  # base 10k photons
            y_noisy = self.rng.poisson(np.clip(counts, 0, None)).astype(np.float64) / (1e4 * pcf) * y_max

        return y_noisy.astype(y.dtype)

    # ------------------------------------------------------------------
    # Shape matching for metric computation
    # ------------------------------------------------------------------

    @staticmethod
    def _match_shapes(
        x_true: np.ndarray,
        x_hat: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Ensure x_true and x_hat have compatible shapes for metric computation.

        When reconstruction produces a lower-dimensional result (e.g., 2-D
        from a 3-D spectral cube), reduce x_true to match by averaging
        along the extra dimension.
        """
        if x_true.shape == x_hat.shape:
            return x_true, x_hat

        # x_hat has fewer dimensions: collapse x_true's extra trailing dims
        if x_hat.ndim < x_true.ndim:
            x_reduced = x_true
            while x_reduced.ndim > x_hat.ndim:
                x_reduced = x_reduced.mean(axis=-1)
            # Also match spatial dimensions via crop/pad
            if x_reduced.shape != x_hat.shape:
                target = x_hat.shape
                out = np.zeros(target, dtype=x_reduced.dtype)
                slices = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(x_reduced.shape, target))
                out[slices] = x_reduced[slices]
                x_reduced = out
            return x_reduced, x_hat

        # x_hat has more dimensions than x_true (unlikely but handle)
        if x_hat.ndim > x_true.ndim:
            x_hat_reduced = x_hat
            while x_hat_reduced.ndim > x_true.ndim:
                x_hat_reduced = x_hat_reduced.mean(axis=-1)
            return x_true, x_hat_reduced

        # Same ndim but different shapes: crop to common region
        common_shape = tuple(min(s1, s2) for s1, s2 in zip(x_true.shape, x_hat.shape))
        slices = tuple(slice(0, s) for s in common_shape)
        return x_true[slices], x_hat[slices]

    # ------------------------------------------------------------------
    # Data resizing
    # ------------------------------------------------------------------

    def _resize_data(self, x_true: np.ndarray, target_shape: List[int]) -> np.ndarray:
        """Crop or pad acquired data to match target image size."""
        target = tuple(target_shape)
        if x_true.shape == target:
            return x_true

        # Build output array
        out = np.zeros(target, dtype=x_true.dtype)

        # Compute overlap region
        slices_src = []
        slices_dst = []
        for i in range(min(x_true.ndim, len(target))):
            size = min(x_true.shape[i], target[i])
            slices_src.append(slice(0, size))
            slices_dst.append(slice(0, size))

        # Handle dimension mismatch
        if x_true.ndim < len(target):
            # Add trailing dimensions
            x_expanded = x_true
            for _ in range(len(target) - x_true.ndim):
                x_expanded = x_expanded[..., np.newaxis]
                slices_src.append(slice(0, 1))
            # Broadcast along new axes
            out[tuple(slices_dst)] = x_expanded[tuple(slices_src)]
            # Tile the last dimension if needed
            if len(target) > x_true.ndim and target[-1] > 1:
                for c in range(1, target[-1]):
                    out[..., c] = out[..., 0]
        elif x_true.ndim > len(target):
            # Squeeze extra dimensions
            x_squeezed = x_true
            for _ in range(x_true.ndim - len(target)):
                x_squeezed = x_squeezed[..., 0]
            slices_src = slices_src[:len(target)]
            out[tuple(slices_dst)] = x_squeezed[tuple(slices_src)]
        else:
            out[tuple(slices_dst)] = x_true[tuple(slices_src)]

        return out

    # ------------------------------------------------------------------
    # Metric set construction
    # ------------------------------------------------------------------

    def _build_metric_set(self, benchmark: str) -> MetricSet:
        """Build a MetricSet from expanded config's bX_config or simple config."""
        bx_config = getattr(self.ecfg, f"{benchmark.lower()}_config", {})
        metric_names = bx_config.get("metrics", None)
        primary = bx_config.get("primary_metric", None)
        thresholds = bx_config.get("thresholds", {})

        # Fall back to simple config
        if not metric_names and self.simple_cfg:
            metric_names = self.simple_cfg.metric_names
            primary = self.simple_cfg.primary_metric
            thresholds = self.simple_cfg.acceptance_thresholds

        return MetricSet(
            names=metric_names or ["psnr", "ssim"],
            primary=primary or "psnr",
            thresholds=thresholds or {},
        )

    # ------------------------------------------------------------------
    # Mismatch helpers
    # ------------------------------------------------------------------

    def _expanded_to_config_mismatch_params(self) -> List[ConfigMismatchParam]:
        """Convert expanded MismatchParam to framework MismatchParam."""
        return [
            ConfigMismatchParam(
                name=mp.name,
                nominal=mp.nominal,
                range=mp.range,
                unit=mp.unit,
            )
            for mp in self.ecfg.mismatch_params
        ]

    def _generate_mismatch_theta(self, case: CaseInstance) -> Dict[str, float]:
        """Generate perturbed theta based on case.mismatch_level."""
        if not case.mismatch_level or not self.ecfg.mismatch_params:
            return {}

        n_perturb = case.mismatch_level.n_params_perturbed
        params = {}

        if n_perturb == 0:
            return {}

        available = list(self.ecfg.mismatch_params)
        if n_perturb == "all" or (isinstance(n_perturb, int) and n_perturb >= len(available)):
            to_perturb = available
        elif isinstance(n_perturb, int):
            to_perturb = available[:n_perturb]
        else:
            to_perturb = available

        for mp in to_perturb:
            params[mp.name] = float(self.rng.uniform(mp.range[0], mp.range[1]))

        return params

    def _true_spec_to_theta(self, case: CaseInstance) -> Dict[str, float]:
        """Convert case.true_spec_params to theta dict for operator building."""
        return dict(case.true_spec_params) if case.true_spec_params else {}

    # ------------------------------------------------------------------
    # Category module helpers
    # ------------------------------------------------------------------

    def _load_category_function(self, fn_name: str) -> Optional[Callable]:
        """Load a function from the category module."""
        cat_module = self.simple_cfg.category_module if self.simple_cfg else ""
        if not cat_module:
            return None
        try:
            mod = importlib.import_module(f"benchmarks.categories.{cat_module}")
            return getattr(mod, fn_name, None)
        except (ImportError, AttributeError):
            return None

    # ------------------------------------------------------------------
    # Result saving
    # ------------------------------------------------------------------

    def _save_case_result(self, result: CaseResult):
        """Save result JSON to results/<modality>/<benchmark>/<variant>/<case_id>.json."""
        result_dir = (
            self.output_dir / result.modality_id / result.benchmark / result.variant_id
        )
        result_dir.mkdir(parents=True, exist_ok=True)
        path = result_dir / f"{result.case_id}.json"
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=_json_default)

    # ------------------------------------------------------------------
    # Parallel execution
    # ------------------------------------------------------------------

    def _run_parallel(
        self,
        cases: List[CaseInstance],
        parallel: int,
        dry_run: bool,
    ) -> List[CaseResult]:
        """Execute cases in parallel using ProcessPoolExecutor."""
        # For parallel execution, we serialize each case and run in subprocesses
        results: List[CaseResult] = []
        self.log(f"Running {len(cases)} cases with {parallel} workers...")

        # ProcessPoolExecutor requires picklable args; run sequentially
        # within each worker using a fresh runner
        futures = {}
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            for case in cases:
                fut = executor.submit(
                    _run_single_case_worker,
                    expanded_config_modality=self.ecfg.modality_id,
                    case_id=case.case_id,
                    benchmark=case.benchmark,
                    case_dict=_case_to_serializable(case),
                    output_dir=str(self.output_dir),
                    seed=self.seed,
                    dry_run=dry_run,
                )
                futures[fut] = case.case_id

            for fut in as_completed(futures):
                cid = futures[fut]
                try:
                    result_dict = fut.result()
                    result = _dict_to_case_result(result_dict)
                    results.append(result)
                    self.log(f"  [OK]   {cid}: {result.status}")
                except Exception as e:
                    result = CaseResult(
                        case_id=cid,
                        status="error",
                        error_message=str(e),
                    )
                    results.append(result)
                    self.log(f"  [FAIL] {cid}: {e}")

        return results

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(self, msg: str):
        if self.verbose:
            print(msg)


# ---------------------------------------------------------------------------
# Parallel worker (module-level for pickling)
# ---------------------------------------------------------------------------

def _run_single_case_worker(
    expanded_config_modality: str,
    case_id: str,
    benchmark: str,
    case_dict: Dict,
    output_dir: str,
    seed: int,
    dry_run: bool,
) -> Dict:
    """Worker function for parallel execution.

    Runs in a subprocess — must re-import everything.
    """
    from benchmarks.framework.expanded_config import load_expanded_config
    from benchmarks.framework.expanded_runner import ExpandedBenchmarkRunner

    ecfg = load_expanded_config(expanded_config_modality)

    # Find the matching case
    gen_fn = getattr(ecfg, f"generate_{benchmark.lower()}_cases")
    target_case = None
    for c in gen_fn():
        if c.case_id == case_id:
            target_case = c
            break

    if target_case is None:
        return {"case_id": case_id, "status": "error", "error_message": "Case not found"}

    # Try loading simple config
    simple_cfg = None
    try:
        cfg_path = Path(__file__).parent.parent / "configs" / f"{expanded_config_modality}.yaml"
        if cfg_path.exists():
            simple_cfg = load_config(cfg_path)
    except Exception:
        pass

    runner = ExpandedBenchmarkRunner(
        expanded_config=ecfg,
        simple_config=simple_cfg,
        output_dir=Path(output_dir),
        verbose=False,
        seed=seed,
    )
    result = runner.run_case(target_case, dry_run=dry_run)
    return result.to_dict()


def _case_to_serializable(case: CaseInstance) -> Dict:
    """Convert a CaseInstance to a minimal serializable dict."""
    return {
        "case_id": case.case_id,
        "modality_id": case.modality_id,
        "benchmark": case.benchmark,
        "variant_id": case.variant.id,
    }


def _dict_to_case_result(d: Dict) -> CaseResult:
    """Reconstruct a CaseResult from a serialized dict."""
    return CaseResult(
        case_id=d.get("case_id", ""),
        modality_id=d.get("modality_id", ""),
        benchmark=d.get("benchmark", ""),
        variant_id=d.get("variant_id", ""),
        status=d.get("status", "error"),
        error_message=d.get("error_message", ""),
        metrics=d.get("metrics", {}),
        wall_time_s=d.get("wall_time_s", 0),
        b1_scores=d.get("b1_scores", {}),
        per_algorithm=d.get("per_algorithm", {}),
        mismatch_results=d.get("mismatch_results", []),
        true_params=d.get("true_params", {}),
        estimated_params=d.get("estimated_params", {}),
        param_errors=d.get("param_errors", {}),
        rho=d.get("rho"),
        grid_search_result=d.get("grid_search_result"),
        psnr_improvement=d.get("psnr_improvement"),
        timestamp=d.get("timestamp", ""),
        run_id=d.get("run_id", ""),
        image_size_id=d.get("image_size_id", ""),
        noise_level_id=d.get("noise_level_id", ""),
        mismatch_level_id=d.get("mismatch_level_id", ""),
    )


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
