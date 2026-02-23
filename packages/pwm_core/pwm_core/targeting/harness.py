"""pwm_core.targeting.harness
==============================

The One Harness: modality-agnostic evaluation engine for LIP-Arena.

Reads from existing YAML registries, compiles OperatorGraphs, runs the
4-scenario protocol, scores with anti-Goodhart penalties, enforces safety
brakes, and emits RunBundles.

This file is rail. It never imports modality-specific code.
"""

from __future__ import annotations

import importlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_operator import GraphOperator
from pwm_core.graph.graph_spec import OperatorGraphSpec
from pwm_core.targeting.budget import BudgetGuard
from pwm_core.targeting.mismatch_sampler import (
    compute_mismatch_magnitude,
    inject_mismatch,
    load_mismatch_db,
    sample_mismatch,
)
from pwm_core.targeting.scenarios import (
    ScenarioResult,
    run_scenario_I,
    run_scenario_II,
    run_scenario_III,
    run_scenario_IV,
)
from pwm_core.targeting.scoring import (
    GateAttribution,
    MaturityLevel,
    ScoredResult,
    compute_maturity_level,
    infer_gate_attribution,
    score_run,
)

logger = logging.getLogger(__name__)

_CONTRIB_DIR = Path(__file__).resolve().parents[2] / "contrib"


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required: pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Registry loaders
# ---------------------------------------------------------------------------


def load_graph_templates(path: Optional[Path] = None) -> Dict[str, Any]:
    path = path or _CONTRIB_DIR / "graph_templates.yaml"
    data = _load_yaml(path)
    return data.get("templates", data)


def load_solver_registry(path: Optional[Path] = None) -> Dict[str, Any]:
    path = path or _CONTRIB_DIR / "solver_registry.yaml"
    return _load_yaml(path)


def load_photon_db(path: Optional[Path] = None) -> Dict[str, Any]:
    path = path or _CONTRIB_DIR / "photon_db.yaml"
    data = _load_yaml(path)
    return data.get("modalities", data)


def load_metrics_db(path: Optional[Path] = None) -> Dict[str, Any]:
    path = path or _CONTRIB_DIR / "metrics_db.yaml"
    return _load_yaml(path)


def _resolve_solver_fn(
    modality: str,
    solver_name: str,
    solver_registry: Dict[str, Any],
) -> Tuple[Callable, Dict[str, Any]]:
    """Resolve a solver name to a callable function.

    Looks up in solver_registry.yaml by modality + tier, or by direct
    module.function specification.
    """
    # Check if solver_name is a tier name
    mod_entry = solver_registry.get(modality, {})
    tier_entry = mod_entry.get(solver_name, {})

    if tier_entry and "module" in tier_entry and "function" in tier_entry:
        module_path = tier_entry["module"]
        func_name = tier_entry["function"]
    elif "." in solver_name:
        # Direct module.function format
        parts = solver_name.rsplit(".", 1)
        module_path, func_name = parts[0], parts[1]
    else:
        # Search across all modalities for matching tier
        for mod, tiers in solver_registry.items():
            if isinstance(tiers, dict):
                for tier, info in tiers.items():
                    if isinstance(info, dict) and info.get("name", "").lower().replace(" ", "_") == solver_name:
                        module_path = info["module"]
                        func_name = info["function"]
                        break
                else:
                    continue
                break
        else:
            raise ValueError(
                f"Solver '{solver_name}' not found in registry for modality "
                f"'{modality}'. Available tiers: {list(mod_entry.keys())}"
            )

    try:
        mod = importlib.import_module(module_path)
        fn = getattr(mod, func_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Cannot load solver '{solver_name}' from "
            f"{module_path}.{func_name}: {e}"
        ) from e

    # Extract optional solver cfg from registry entry
    solver_cfg: Dict[str, Any] = {}
    if isinstance(tier_entry, dict):
        solver_cfg = {k: v for k, v in tier_entry.get("cfg", {}).items()}

    return fn, solver_cfg


def _generate_scene(
    x_shape: Tuple[int, ...],
    rng: np.random.Generator,
    modality: str = "",
) -> np.ndarray:
    """Generate a synthetic scene for evaluation.

    For CT, returns a Gaussian-smoothed phantom with sharp object boundaries
    so that TV-based calibration metrics are meaningful.  Other modalities use
    normalised Gaussian noise (their calibration metrics are noise-agnostic).
    """
    if modality in ("ct", "cbct"):
        from scipy.ndimage import gaussian_filter
        H = x_shape[0] if len(x_shape) >= 1 else 32
        W = x_shape[1] if len(x_shape) >= 2 else H
        # Smooth blobs: Gaussian-blur random noise → piecewise-smooth phantom
        raw = rng.standard_normal((H, W)).astype(np.float64)
        blurred = gaussian_filter(raw, sigma=max(H // 8, 2))
        # Threshold to create disc-like high-contrast regions
        x = (blurred > 0.0).astype(np.float64)
        # Add a smoothly-graded interior
        x = x + 0.3 * gaussian_filter(raw, sigma=max(H // 4, 3))
        x = np.clip(x, 0.0, None)
        x = (x - x.min()) / (x.max() - x.min() + 1e-10)
        return x
    x = rng.standard_normal(x_shape).astype(np.float64)
    x = (x - x.min()) / (x.max() - x.min() + 1e-10)
    return x


def _add_noise(
    y_clean: np.ndarray,
    photon_config: Dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """Add noise to measurements based on photon_db config."""
    noise_model = photon_config.get("noise_model", "gaussian")
    levels = photon_config.get("photon_levels", {})
    level = levels.get("standard", levels.get("bright", {}))
    n_photons = level.get("n_photons", 10000)
    read_sigma = level.get("read_sigma_fraction", 0.01)

    y = y_clean.copy()
    if noise_model in ("poisson_gaussian", "poisson"):
        # Scale to photon counts, apply Poisson, scale back
        scale = float(n_photons) / (y.max() + 1e-10)
        y_photons = rng.poisson(np.clip(y * scale, 0, None).astype(np.float64))
        y = y_photons.astype(np.float64) / scale
    if noise_model in ("poisson_gaussian", "gaussian"):
        if np.iscomplexobj(y):
            # Complex measurements (e.g. MRI k-space): add independent real + imag noise
            sigma = read_sigma * (np.abs(y).max() - np.abs(y).min() + 1e-10)
            y = y + rng.normal(0, sigma, y.shape) + 1j * rng.normal(0, sigma, y.shape)
        else:
            sigma = read_sigma * (y.max() - y.min() + 1e-10)
            y = y + rng.normal(0, sigma, y.shape)
    return y


# ---------------------------------------------------------------------------
# DecisionRecord — DR-AIS (Decision Records for AI Systems)
# ---------------------------------------------------------------------------


@dataclass
class DecisionRecord:
    """Immutable audit log for a harness run (DR-AIS standard).

    Implements the Decision Records for AI Systems concept from the
    SolveEverything Industrial Intelligence Stack.  Records exactly what
    was measured, which physics model was used, and why the dominant gate
    was attributed — functioning as a flight recorder for the Targeting System.

    Attributes
    ----------
    run_id : str
        Unique identifier for this evaluation run.
    modality : str
        Imaging modality evaluated.
    solver : str
        Solver tier used.
    maturity_label : str
        L0-L5 label for the aggregate run (e.g. "L3").
    maturity_name : str
        Human-readable maturity name (e.g. "Automated").
    rho : float
        Aggregate recovery ratio.
    dominant_gate : str
        Dominant Triad gate identified.
    recommended_action : str
        Actionable recommendation from the dominant gate.
    blinded_clear : bool
        True when mismatch was injected blind (as per CASP protocol).
    template_id : str
        Graph template used for the physics operator.
    severity : str
        Mismatch severity level used ('mild', 'moderate', 'severe', 'catastrophic').
    n_scenes : int
        Number of scenes evaluated.
    """

    run_id: str
    modality: str
    solver: str
    maturity_label: str
    maturity_name: str
    rho: float
    dominant_gate: str
    recommended_action: str
    blinded_clear: bool
    template_id: str
    severity: str
    n_scenes: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "modality": self.modality,
            "solver": self.solver,
            "maturity_label": self.maturity_label,
            "maturity_name": self.maturity_name,
            "rho": self.rho,
            "dominant_gate": self.dominant_gate,
            "recommended_action": self.recommended_action,
            "blinded_clear": self.blinded_clear,
            "template_id": self.template_id,
            "severity": self.severity,
            "n_scenes": self.n_scenes,
        }

    def __str__(self) -> str:
        return (
            f"DR-AIS | {self.modality}/{self.solver} | "
            f"{self.maturity_label} ({self.maturity_name}) | "
            f"ρ={self.rho:.4f} | gate={self.dominant_gate} | "
            f"blinded={self.blinded_clear}"
        )


# ---------------------------------------------------------------------------
# HarnessResult
# ---------------------------------------------------------------------------


@dataclass
class SceneResult:
    """Results for one scene."""

    scene_idx: int
    scenario_results: Dict[str, ScenarioResult]
    scored: ScoredResult
    mismatch_params: Dict[str, float]
    mismatch_magnitude: float
    gate_attribution: Optional[GateAttribution] = None


@dataclass
class HarnessResult:
    """Complete results for a harness run.

    Includes LIP-Arena metrics plus SolveEverything Industrial Intelligence
    Stack indicators: L0-L5 maturity classification and DR-AIS decision record.
    """

    modality: str
    solver: str
    track: str
    n_scenes: int
    seed: int
    severity: str
    sandbox: bool
    per_scene: List[SceneResult]
    aggregate: ScoredResult
    timing_s: float
    budget_report: Dict[str, Any]
    gate_attribution: Optional[GateAttribution] = None
    runbundle_path: Optional[str] = None
    # SolveEverything Industrial Intelligence Stack additions
    maturity_level: Optional[MaturityLevel] = None
    decision_record: Optional[DecisionRecord] = None

    def summary_table(self) -> str:
        """Return a formatted summary table."""
        # Build maturity label for header
        ml = self.maturity_level or (
            self.aggregate.maturity_level
            if self.aggregate.maturity_level is not None
            else None
        )
        ml_str = f"{ml.label} ({ml.name})" if ml is not None else "—"

        lines = [
            f"{'='*60}",
            f"  LIP-Arena Targeting System  [Industrial Intelligence Stack L4]",
            f"  Modality: {self.modality}  |  Solver: {self.solver}",
            f"  Track: {self.track}  |  Scenes: {self.n_scenes}",
            f"  Severity: {self.severity}  |  Sandbox: {self.sandbox}",
            f"  Maturity: {ml_str}",
            f"{'='*60}",
            f"",
            f"  {'Metric':<25} {'Value':>10}",
            f"  {'-'*35}",
            f"  {'Recovery ratio (rho)':<25} {self.aggregate.rho:>10.4f}",
            f"  {'Oracle gap (dB)':<25} {self.aggregate.oracle_gap:>10.2f}",
            f"  {'RoIC (dB/GPU-hr)':<25} {self.aggregate.roic:>10.2f}",
            f"  {'RoCS (dB/budget-frac)':<25} {self.aggregate.rocs:>10.2f}",
            f"  {'OFS':<25} {self.aggregate.ofs:>10.4f}",
            f"  {'Final score':<25} {self.aggregate.final_score:>10.4f}",
            f"  {'-'*35}",
            f"  {'Total time (s)':<25} {self.timing_s:>10.2f}",
        ]

        if self.gate_attribution is not None:
            ga = self.gate_attribution
            lines += [
                f"",
                f"  Triad Gate Attribution",
                f"  {'-'*35}",
                f"  {'Gate 1 (Sampling)':<25} {ga.gate_1_sampling:>10.3f}",
                f"  {'Gate 2 (Noise)':<25} {ga.gate_2_noise:>10.3f}",
                f"  {'Gate 3 (Mismatch)':<25} {ga.gate_3_mismatch:>10.3f}",
                f"  {'Dominant gate':<25} {ga.dominant_gate:>10}",
                f"  {'Confidence':<25} {ga.confidence:>10.3f}",
            ]

        if self.decision_record is not None:
            dr = self.decision_record
            lines += [
                f"",
                f"  DR-AIS Decision Record",
                f"  {'-'*35}",
                f"  {'Run ID':<25} {dr.run_id:>10}",
                f"  {'Blinded clear':<25} {'yes' if dr.blinded_clear else 'no':>10}",
                f"  {'Template':<25} {dr.template_id:>10}",
            ]

        if self.aggregate.disqualified:
            lines.append(f"  ** DISQUALIFIED: {self.aggregate.disqualification_reason}")

        if self.aggregate.safety_violations:
            lines.append(f"  ** SAFETY BRAKES: {len(self.aggregate.safety_violations)} triggered")
            for v in self.aggregate.safety_violations:
                lines.append(f"     - {v.condition}: {v.action}")

        if self.aggregate.penalties:
            lines.append(f"  ** PENALTIES: {len(self.aggregate.penalties)} applied")
            for p in self.aggregate.penalties:
                lines.append(f"     - {p.check}: -{p.penalty_fraction*100:.0f}%")

        lines.append(f"{'='*60}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        ml = self.maturity_level or (
            self.aggregate.maturity_level
            if self.aggregate.maturity_level is not None
            else None
        )
        return {
            "modality": self.modality,
            "solver": self.solver,
            "track": self.track,
            "n_scenes": self.n_scenes,
            "seed": self.seed,
            "severity": self.severity,
            "sandbox": self.sandbox,
            "aggregate": self.aggregate.to_dict(),
            "maturity_level": ml.to_dict() if ml is not None else None,
            "decision_record": (
                self.decision_record.to_dict()
                if self.decision_record is not None else None
            ),
            "gate_attribution": (
                self.gate_attribution.to_dict()
                if self.gate_attribution is not None else None
            ),
            "timing_s": self.timing_s,
            "budget_report": self.budget_report,
            "per_scene": [
                {
                    "scene_idx": s.scene_idx,
                    "rho": s.scored.rho,
                    "oracle_gap": s.scored.oracle_gap,
                    "roic": s.scored.roic,
                    "mismatch_magnitude": s.mismatch_magnitude,
                    "gate_attribution": (
                        s.gate_attribution.to_dict()
                        if s.gate_attribution is not None else None
                    ),
                    "scenarios": {
                        k: {"psnr": v.psnr, "ssim": v.ssim, "runtime_s": v.runtime_s}
                        for k, v in s.scenario_results.items()
                    },
                }
                for s in self.per_scene
            ],
        }


# ---------------------------------------------------------------------------
# The Harness
# ---------------------------------------------------------------------------


class Harness:
    """The One Harness: modality-agnostic evaluation engine.

    Parameters
    ----------
    modality : str
        Modality name (looked up in graph_templates.yaml, mismatch_db.yaml, etc.).
    solver : str
        Solver name or tier (looked up in solver_registry.yaml).
    track : str
        Evaluation track: 'correct', 'diagnose', 'no_gt', 'design'.
    budget_s : int
        Compute budget in seconds.
    sandbox : bool
        If True, use tiny operators, 1 scene, no escrow enforcement.
    calibrator_fn : callable, optional
        Calibrator function for Scenario III. If None, Scenario III = Scenario II.
    solver_fn : callable, optional
        Override: directly pass the solver function instead of loading from registry.
    """

    def __init__(
        self,
        modality: str,
        solver: str = "traditional_cpu",
        track: str = "correct",
        budget_s: int = 600,
        sandbox: bool = False,
        calibrator_fn: Optional[Callable] = None,
        solver_fn: Optional[Callable] = None,
    ):
        self.modality = modality
        self.solver_name = solver
        self.track = track
        self.budget_s = budget_s
        self.sandbox = sandbox

        # Auto-load default calibrator when none supplied
        if calibrator_fn is None:
            from pwm_core.targeting.calibrators import get_default_calibrator
            calibrator_fn = get_default_calibrator(modality)
        self.calibrator_fn = calibrator_fn

        # Load registries
        self.graph_templates = load_graph_templates()
        self.mismatch_db = load_mismatch_db()
        self.solver_registry = load_solver_registry()
        self.photon_db = load_photon_db()

        # Resolve graph template
        template_id = f"{modality}_graph_v1"
        if template_id not in self.graph_templates:
            available = [k for k in self.graph_templates if k.startswith(modality)]
            if available:
                template_id = available[0]
            else:
                raise ValueError(
                    f"No graph template for modality '{modality}'. "
                    f"Available: {sorted(self.graph_templates.keys())}"
                )
        self.template = self.graph_templates[template_id]
        self.template_id = template_id

        # Resolve solver function
        if solver_fn is not None:
            self._solver_fn = solver_fn
            self._solver_cfg: Dict[str, Any] = {}
        else:
            self._solver_fn, self._solver_cfg = _resolve_solver_fn(
                modality, solver, self.solver_registry
            )

        # Compile operators
        self.compiler = GraphCompiler()

        logger.info(
            f"Harness initialized: modality={modality}, solver={solver}, "
            f"track={track}, budget={budget_s}s, sandbox={sandbox}"
        )

    def _compile_operator(
        self, x_shape: Optional[Tuple[int, ...]] = None
    ) -> GraphOperator:
        """Compile the OperatorGraph for this modality.

        Noise nodes are stripped from the graph because the harness adds
        measurement noise independently via ``_add_noise()``.  This keeps
        the compiled operator fully linear so ``.adjoint()`` works.
        """
        # OperatorGraphSpec uses extra="forbid", so strip fields that aren't
        # part of the Pydantic model (e.g. 'description' from YAML templates).
        allowed_keys = {"graph_id", "nodes", "edges", "metadata"}
        spec_data = {"graph_id": self.template_id}
        for k, v in self.template.items():
            if k in allowed_keys:
                spec_data[k] = v

        # Strip noise nodes and their edges so the operator is linear
        if "nodes" in spec_data:
            noise_ids = {
                n["node_id"]
                for n in spec_data["nodes"]
                if n.get("role") == "noise"
                or n.get("primitive_id", "").startswith("noise")
                or "noise" in n.get("node_id", "").lower()
            }
            if noise_ids:
                spec_data["nodes"] = [
                    n for n in spec_data["nodes"]
                    if n["node_id"] not in noise_ids
                ]
                if "edges" in spec_data:
                    spec_data["edges"] = [
                        e for e in spec_data["edges"]
                        if e.get("source") not in noise_ids
                        and e.get("target") not in noise_ids
                    ]
                # Canonical-chain validation requires a noise node, which we
                # just stripped.  Disable the check so the compiler doesn't
                # reject the now-linear graph.
                if "metadata" in spec_data:
                    spec_data["metadata"] = dict(spec_data["metadata"])
                    spec_data["metadata"].pop("canonical_chain", None)

        spec = GraphCompiler.from_dict(spec_data)

        # Sandbox: override to tiny shapes and propagate to primitive params
        if self.sandbox:
            meta = spec.metadata.copy()
            orig_x = tuple(meta.get("x_shape", [64, 64]))
            # Shrink to max 32 per dimension
            sandbox_x = tuple(min(d, 32) for d in orig_x)
            sandbox_y = tuple(min(d, 32) for d in tuple(meta.get("y_shape", list(orig_x))))
            x_shape = x_shape or sandbox_x

            # Propagate sandbox spatial dims to primitive params that
            # hardcode H/W (e.g. coded_mask). Without this, the mask
            # shape (64x64) mismatches the sandbox scene shape (32x32).
            sb_h, sb_w = x_shape[0], x_shape[1] if len(x_shape) > 1 else x_shape[0]
            for node in spec.nodes:
                if node.params:
                    if "H" in node.params:
                        node.params["H"] = sb_h
                    if "W" in node.params:
                        node.params["W"] = sb_w

            return self.compiler.compile(spec, x_shape=x_shape, y_shape=sandbox_y)

        return self.compiler.compile(spec, x_shape=x_shape)

    def run(
        self,
        n_scenes: int = 5,
        seed: int = 42,
        severity: str = "moderate",
    ) -> HarnessResult:
        """Run the full evaluation protocol.

        Parameters
        ----------
        n_scenes : int
            Number of synthetic scenes to evaluate.
        seed : int
            Random seed for reproducibility.
        severity : str
            Mismatch severity: 'mild', 'moderate', 'severe', 'catastrophic'.

        Returns
        -------
        HarnessResult
            Complete results with per-scene and aggregate scores.
        """
        if self.sandbox:
            n_scenes = 1

        t_start = time.perf_counter()
        rng = np.random.default_rng(seed)
        budget_guard = BudgetGuard(
            budget_s=self.budget_s,
            enforce=not self.sandbox,
        )

        scene_results: List[SceneResult] = []

        for scene_idx in range(n_scenes):
            scene_rng = np.random.default_rng(rng.integers(0, 2**31))

            # 1. Compile H_true
            H_true = self._compile_operator()

            # 2. Generate synthetic scene
            x_true = _generate_scene(H_true.x_shape, scene_rng, modality=self.modality)

            # 3. Sample mismatch -> create H_nom
            theta_mismatch = sample_mismatch(
                self.modality, severity, scene_rng, self.mismatch_db
            )
            H_nom = inject_mismatch(H_true, theta_mismatch)
            mismatch_mag = compute_mismatch_magnitude(theta_mismatch)

            # 4. Generate measurement: y = H_true(x) + noise
            y_clean = H_true.forward(x_true)
            photon_config = self.photon_db.get(self.modality, {})
            y = _add_noise(y_clean, photon_config, scene_rng)

            # 5. Run 4 scenarios
            # Use solver_cfg from registry as default; callers can override.
            cfg: Dict[str, Any] = dict(self._solver_cfg)
            scenarios: Dict[str, ScenarioResult] = {}

            scenarios["I"] = run_scenario_I(
                x_true, y, H_true, self._solver_fn, cfg, budget_guard
            )
            scenarios["II"] = run_scenario_II(
                x_true, y, H_nom, self._solver_fn, cfg, budget_guard
            )
            scenarios["III"] = run_scenario_III(
                x_true, y, H_nom, self.calibrator_fn,
                self._solver_fn, cfg, budget_guard,
            )
            scenarios["IV"] = run_scenario_IV(
                x_true, y, H_true, H_nom, self._solver_fn, cfg, budget_guard
            )

            # 6. Score
            total_time = sum(s.runtime_s for s in scenarios.values())
            scored = score_run(
                psnr_i=scenarios["I"].psnr,
                psnr_ii=scenarios["II"].psnr,
                psnr_iii=scenarios["III"].psnr,
                psnr_iv=scenarios["IV"].psnr,
                total_gpu_seconds=total_time,
                track=self.track,
                mismatch_magnitude=mismatch_mag,
                budget_s=float(self.budget_s),
            )

            # 6b. Gate attribution (Triad decomposition for Track 2)
            gate_attr = infer_gate_attribution(
                psnr_i=scenarios["I"].psnr,
                psnr_ii=scenarios["II"].psnr,
                psnr_iii=scenarios["III"].psnr,
                psnr_iv=scenarios["IV"].psnr,
                mismatch_magnitude=mismatch_mag,
            )

            scene_results.append(SceneResult(
                scene_idx=scene_idx,
                scenario_results=scenarios,
                scored=scored,
                mismatch_params=theta_mismatch,
                mismatch_magnitude=mismatch_mag,
                gate_attribution=gate_attr,
            ))

            logger.info(
                f"Scene {scene_idx}: rho={scored.rho:.4f}, "
                f"oracle_gap={scored.oracle_gap:.2f} dB, "
                f"dominant_gate={gate_attr.dominant_gate}, "
                f"roic={scored.roic:.2f}"
            )

        # 7. Aggregate across scenes
        avg_rho = float(np.mean([s.scored.rho for s in scene_results]))
        avg_gap = float(np.mean([s.scored.oracle_gap for s in scene_results]))
        avg_roic = float(np.mean([s.scored.roic for s in scene_results]))
        avg_ofs = float(np.mean([s.scored.ofs for s in scene_results]))
        avg_score = float(np.mean([s.scored.final_score for s in scene_results]))

        # Collect all safety violations and penalties
        all_violations = []
        all_penalties = []
        any_dq = False
        dq_reason = ""
        for s in scene_results:
            all_violations.extend(s.scored.safety_violations)
            all_penalties.extend(s.scored.penalties)
            if s.scored.disqualified:
                any_dq = True
                dq_reason = s.scored.disqualification_reason

        aggregate = ScoredResult(
            rho=avg_rho,
            oracle_gap=avg_gap,
            roic=avg_roic,
            ofs=avg_ofs,
            track=self.track,
            track_score=avg_score,
            penalties=all_penalties,
            safety_violations=all_violations,
            final_score=0.0 if any_dq else avg_score,
            disqualified=any_dq,
            disqualification_reason=dq_reason,
        )

        # 7b. Aggregate gate attribution across scenes
        scene_gas = [
            s.gate_attribution for s in scene_results if s.gate_attribution is not None
        ]
        agg_gate_attr: Optional[GateAttribution] = None
        if scene_gas:
            avg_g1 = float(np.mean([g.gate_1_sampling for g in scene_gas]))
            avg_g2 = float(np.mean([g.gate_2_noise for g in scene_gas]))
            avg_g3 = float(np.mean([g.gate_3_mismatch for g in scene_gas]))
            avg_conf = float(np.mean([g.confidence for g in scene_gas]))
            avg_fracs = {
                "gate_1_sampling": avg_g1,
                "gate_2_noise": avg_g2,
                "gate_3_mismatch": avg_g3,
            }
            agg_dominant = max(avg_fracs, key=avg_fracs.get)  # type: ignore[arg-type]
            from pwm_core.targeting.scoring import _GATE_ACTIONS
            agg_gate_attr = GateAttribution(
                gate_1_sampling=round(avg_g1, 4),
                gate_2_noise=round(avg_g2, 4),
                gate_3_mismatch=round(avg_g3, 4),
                dominant_gate=agg_dominant,
                confidence=round(avg_conf, 4),
                recommended_action=_GATE_ACTIONS[agg_dominant],
                evidence={"n_scenes": len(scene_gas)},
            )

        total_time = time.perf_counter() - t_start
        budget_report = budget_guard.report().to_dict()

        # SolveEverything: L0-L5 maturity level for the aggregate run
        agg_maturity = compute_maturity_level(avg_rho)

        # DR-AIS: immutable decision record for this run
        agg_dom_gate = agg_gate_attr.dominant_gate if agg_gate_attr else "unknown"
        agg_rec_action = agg_gate_attr.recommended_action if agg_gate_attr else ""
        decision_record = DecisionRecord(
            run_id=str(uuid.uuid4())[:8],
            modality=self.modality,
            solver=self.solver_name,
            maturity_label=agg_maturity.label,
            maturity_name=agg_maturity.name,
            rho=round(avg_rho, 4),
            dominant_gate=agg_dom_gate,
            recommended_action=agg_rec_action,
            blinded_clear=True,  # mismatch always injected blind
            template_id=self.template_id,
            severity=severity,
            n_scenes=n_scenes,
        )

        result = HarnessResult(
            modality=self.modality,
            solver=self.solver_name,
            track=self.track,
            n_scenes=n_scenes,
            seed=seed,
            severity=severity,
            sandbox=self.sandbox,
            per_scene=scene_results,
            aggregate=aggregate,
            gate_attribution=agg_gate_attr,
            timing_s=total_time,
            budget_report=budget_report,
            maturity_level=agg_maturity,
            decision_record=decision_record,
        )

        logger.info(
            f"Harness complete: rho={avg_rho:.4f}, "
            f"oracle_gap={avg_gap:.2f} dB, "
            f"final_score={aggregate.final_score:.4f}, "
            f"time={total_time:.1f}s"
        )

        return result
