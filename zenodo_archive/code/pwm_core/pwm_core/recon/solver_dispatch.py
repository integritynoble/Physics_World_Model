"""Unified solver dispatch for all PWM use cases.

Three deployment modes:
1. Server mode: Main server (no GPU) calls CPU solvers directly, GPU solvers via Modal
2. Paper mode: Run solvers for benchmark tables in papers
3. CLI mode: `pwm evaluate --method <solver> --modality <modality> --track correct`

Usage:
    from pwm_core.recon.solver_dispatch import SolverDispatcher

    dispatcher = SolverDispatcher()

    # Run a specific solver
    x_hat = dispatcher.run(modality_id="cassi", solver_key="traditional_cpu",
                           y=measurement, operator=op)

    # Run best solver for a modality
    x_hat = dispatcher.run_best(modality_id="cassi", y=measurement, operator=op)

    # List available solvers
    solvers = dispatcher.list_solvers("cassi")

    # Check if solver needs GPU
    needs_gpu = dispatcher.needs_gpu("cassi", "best_quality")
"""

from __future__ import annotations

import importlib
import json
import time
import yaml
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class SolverDispatcher:
    """Unified solver dispatch for server, paper, and CLI usage."""

    def __init__(self, root: Optional[Path] = None, use_modal: bool = False):
        """Initialize dispatcher.

        Args:
            root: Project root directory. Auto-detected if None.
            use_modal: If True, GPU solvers are dispatched to Modal.
                       If False, GPU solvers run locally (requires local GPU).
        """
        if root is None:
            # Auto-detect from this file's location
            root = Path(__file__).parent.parent.parent.parent.parent
        self.root = Path(root)
        self.config_dir = self.root / "benchmarks" / "configs"
        self.use_modal = use_modal
        self._config_cache: Dict[str, dict] = {}
        self._registry_cache: Optional[dict] = None

    def _load_config(self, modality_id: str) -> dict:
        """Load modality YAML config."""
        if modality_id in self._config_cache:
            return self._config_cache[modality_id]
        config_path = self.config_dir / f"{modality_id}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self._config_cache[modality_id] = cfg
        return cfg

    def _load_registry(self) -> dict:
        """Load solver registry."""
        if self._registry_cache is not None:
            return self._registry_cache
        reg_path = self.root / "benchmark_results" / "solver_registry.json"
        if reg_path.exists():
            with open(reg_path, "r", encoding="utf-8") as f:
                self._registry_cache = json.load(f)
        else:
            self._registry_cache = {"modalities": {}}
        return self._registry_cache

    def list_solvers(self, modality_id: str) -> List[Dict[str, Any]]:
        """List all available solvers for a modality.

        Returns:
            List of solver specs with keys: id, name, module, function, gpu, reference
        """
        cfg = self._load_config(modality_id)
        solvers = cfg.get("solvers", {})
        result = []
        for sk, spec in solvers.items():
            if not isinstance(spec, dict):
                continue
            result.append({
                "id": sk,
                "name": spec.get("name", sk),
                "module": spec.get("module", ""),
                "function": spec.get("function", ""),
                "gpu": spec.get("gpu", False),
                "reference": spec.get("reference", ""),
                "params": spec.get("params", {}),
            })
        return result

    def needs_gpu(self, modality_id: str, solver_key: str) -> bool:
        """Check if a solver requires GPU."""
        cfg = self._load_config(modality_id)
        solvers = cfg.get("solvers", {})
        spec = solvers.get(solver_key, {})
        return spec.get("gpu", False) if isinstance(spec, dict) else False

    def get_solver_spec(self, modality_id: str, solver_key: str) -> Dict[str, Any]:
        """Get full solver specification."""
        cfg = self._load_config(modality_id)
        solvers = cfg.get("solvers", {})
        spec = solvers.get(solver_key)
        if spec is None or not isinstance(spec, dict):
            raise ValueError(f"Solver '{solver_key}' not found in '{modality_id}'")
        return {
            "id": solver_key,
            "name": spec.get("name", solver_key),
            "module": spec.get("module", ""),
            "function": spec.get("function", ""),
            "gpu": spec.get("gpu", False),
            "reference": spec.get("reference", ""),
            "params": spec.get("params", {}),
        }

    def run(
        self,
        modality_id: str,
        solver_key: str,
        y: np.ndarray,
        operator: Any = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run a solver.

        Args:
            modality_id: Modality identifier
            solver_key: Solver key from YAML config
            y: Measurement data
            operator: Forward operator (optional, depends on solver)
            params: Override solver parameters

        Returns:
            Tuple of (x_hat, metadata) where metadata includes time, solver info
        """
        spec = self.get_solver_spec(modality_id, solver_key)
        gpu_required = spec["gpu"]

        if gpu_required and self.use_modal:
            return self._run_modal(modality_id, solver_key, y, operator, params, spec)
        else:
            return self._run_local(modality_id, solver_key, y, operator, params, spec)

    def _run_local(
        self,
        modality_id: str,
        solver_key: str,
        y: np.ndarray,
        operator: Any,
        params: Optional[Dict[str, Any]],
        spec: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run solver locally."""
        module_name = spec["module"]
        func_name = spec["function"]
        solver_params = params or spec.get("params", {})
        if isinstance(solver_params, str):
            solver_params = {}

        t0 = time.time()
        try:
            mod = importlib.import_module(module_name)
            fn = getattr(mod, func_name)
            result = fn(y.astype(np.float32), operator, solver_params)
            if isinstance(result, tuple):
                x_hat = np.asarray(result[0], dtype=np.float32)
            else:
                x_hat = np.asarray(result, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(
                f"Solver {solver_key} ({module_name}.{func_name}) failed: {e}"
            ) from e

        elapsed = time.time() - t0
        metadata = {
            "solver_key": solver_key,
            "solver_name": spec["name"],
            "module": module_name,
            "function": func_name,
            "gpu": spec["gpu"],
            "time": round(elapsed, 3),
            "dispatch": "local",
        }
        return x_hat, metadata

    def _run_modal(
        self,
        modality_id: str,
        solver_key: str,
        y: np.ndarray,
        operator: Any,
        params: Optional[Dict[str, Any]],
        spec: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run solver on Modal GPU."""
        import base64

        try:
            from benchmarks.modal.gpu_solver import run_gpu_solver
        except ImportError:
            raise RuntimeError(
                "Modal not available. Install with: pip install modal\n"
                "Or set use_modal=False for local GPU execution."
            )

        # Prepare operator data if available
        operator_data = None
        operator_shape = None
        if operator is not None and hasattr(operator, "A") and operator.A is not None:
            A = np.asarray(operator.A, dtype=np.float32)
            operator_data = A.tobytes()
            operator_shape = list(A.shape)

        result = run_gpu_solver.remote(
            modality_id=modality_id,
            solver_key=solver_key,
            y_data=y.astype(np.float32).tobytes(),
            y_shape=list(y.shape),
            y_dtype="float32",
            operator_data=operator_data,
            operator_shape=operator_shape,
            params=params,
        )

        if "error" in result:
            raise RuntimeError(f"Modal solver failed: {result['error']}")

        x_bytes = base64.b64decode(result["x_hat"])
        x_hat = np.frombuffer(x_bytes, dtype=result["dtype"]).reshape(result["shape"])

        metadata = {
            "solver_key": solver_key,
            "solver_name": spec["name"],
            "module": spec["module"],
            "function": spec["function"],
            "gpu": True,
            "time": result.get("time", 0),
            "dispatch": "modal",
        }
        return x_hat.copy(), metadata

    def run_best(
        self,
        modality_id: str,
        y: np.ndarray,
        operator: Any = None,
        prefer_cpu: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run the best available solver for a modality.

        Args:
            modality_id: Modality identifier
            y: Measurement data
            operator: Forward operator
            prefer_cpu: If True, prefer CPU solvers over GPU when quality is similar

        Returns:
            Tuple of (x_hat, metadata)
        """
        solvers = self.list_solvers(modality_id)

        # Priority order for solver selection
        priority = ["best_quality", "famous_dl", "traditional_cpu"]

        if prefer_cpu:
            # Try CPU solvers first
            cpu_solvers = [s for s in solvers if not s["gpu"]]
            for tier in priority:
                for s in cpu_solvers:
                    if s["id"] == tier:
                        return self.run(modality_id, s["id"], y, operator)
            # Fallback to any CPU solver
            if cpu_solvers:
                return self.run(modality_id, cpu_solvers[0]["id"], y, operator)

        # Try in priority order (any solver)
        for tier in priority:
            for s in solvers:
                if s["id"] == tier:
                    return self.run(modality_id, s["id"], y, operator)

        # Fallback to first available
        if solvers:
            return self.run(modality_id, solvers[0]["id"], y, operator)

        raise ValueError(f"No solvers available for '{modality_id}'")

    def run_all(
        self,
        modality_id: str,
        y: np.ndarray,
        operator: Any = None,
        cpu_only: bool = False,
    ) -> List[Tuple[str, np.ndarray, Dict[str, Any]]]:
        """Run all available solvers for a modality.

        Args:
            modality_id: Modality identifier
            y: Measurement data
            operator: Forward operator
            cpu_only: If True, skip GPU solvers

        Returns:
            List of (solver_key, x_hat, metadata) tuples
        """
        solvers = self.list_solvers(modality_id)
        results = []

        for s in solvers:
            if cpu_only and s["gpu"]:
                continue
            try:
                x_hat, meta = self.run(modality_id, s["id"], y, operator)
                results.append((s["id"], x_hat, meta))
            except Exception as e:
                results.append((s["id"], None, {"error": str(e)}))

        return results

    @staticmethod
    def list_modalities(config_dir: Optional[Path] = None) -> List[str]:
        """List all available modality IDs."""
        if config_dir is None:
            config_dir = Path(__file__).parent.parent.parent.parent.parent / "benchmarks" / "configs"
        mods = []
        for f in sorted(config_dir.iterdir()):
            if f.suffix == ".yaml" and f.stem != "_template":
                mods.append(f.stem)
        return mods
