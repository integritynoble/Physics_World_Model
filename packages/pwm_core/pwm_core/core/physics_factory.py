"""pwm_core.core.physics_factory

Factory for building physics operators from ExperimentSpec.

Routes to appropriate operator based on modality:
- widefield: WidefieldOperator (Gaussian blur)
- confocal: WidefieldOperator (sharper PSF)
- sim: SIMOperator (structured illumination)
- cassi: CASSIOperator (coded aperture spectral imaging)
- spc: SPCOperator (single-pixel camera)
- cacti: CACTIOperator (video snapshot compressive imaging)
- lensless: LenslessOperator (diffuser camera)
- lightsheet: LightsheetOperator (light-sheet microscopy)
- ct: CTOperator (computed tomography)
- mri: MRIOperator (MRI k-space)
- ptychography: PtychographyOperator (ptychographic imaging)
- holography: HolographyOperator (off-axis holography)
- nerf: NeRFOperator (neural radiance fields)
- gaussian_splatting: GaussianSplattingOperator (3D Gaussian splatting)
- oct: OCTOperator (optical coherence tomography)
- light_field: LightFieldOperator (microlens array light field)
- matrix: MatrixOperator (explicit matrix A)
- callable: CallableOperator (user-provided forward/adjoint)
- identity: fallback for testing
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from pwm_core.api.types import ExperimentSpec, OperatorKind
from pwm_core.physics.base import BaseOperator

logger = logging.getLogger(__name__)


class IdentityOperator(BaseOperator):
    """Identity operator for testing/fallback."""

    def __init__(self, x_shape: Tuple[int, ...] = (64, 64)):
        self.operator_id = "identity"
        self.theta = {}
        self.x_shape = x_shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x.astype(np.float32)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        return y.astype(np.float32)

    def info(self) -> Dict[str, Any]:
        return {"operator_id": self.operator_id, "x_shape": self.x_shape}


def _get_dims_from_spec(spec: ExperimentSpec) -> Tuple[int, ...]:
    """Extract dimensions from spec, with fallback defaults."""
    dims = spec.states.physics.dims
    if dims is None:
        return (64, 64)

    # Handle various dims formats
    if isinstance(dims, dict):
        h = dims.get('H', dims.get('height', 64))
        w = dims.get('W', dims.get('width', 64))
        d = dims.get('D', dims.get('depth', None))
        l = dims.get('L', dims.get('bands', None))
        if d is not None:
            return (int(h), int(w), int(d))
        if l is not None:
            return (int(h), int(w), int(l))
        return (int(h), int(w))
    elif isinstance(dims, (list, tuple)):
        return tuple(int(d) for d in dims)

    return (64, 64)


# Stochastic primitive IDs that should be stripped from graph templates
# before compiling the reconstruction operator. Noise is handled separately
# by simulator.py.
_STOCHASTIC_PRIMITIVE_IDS = frozenset({"poisson", "gaussian", "poisson_gaussian", "fpn"})


def _load_graph_templates() -> Dict[str, Any]:
    """Load graph_templates.yaml from the contrib directory."""
    contrib_dir = os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, "contrib"
    )
    templates_path = os.path.normpath(
        os.path.join(contrib_dir, "graph_templates.yaml")
    )
    if not os.path.exists(templates_path):
        return {}
    with open(templates_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("templates", {}) if data else {}


def _strip_stochastic_nodes(template: Dict[str, Any]) -> Dict[str, Any]:
    """Remove stochastic nodes and their edges from a graph template.

    Returns a copy with stochastic nodes (noise primitives) removed,
    since noise is applied separately by simulator.py.
    """
    template = dict(template)  # shallow copy
    nodes = list(template.get("nodes", []))
    edges = list(template.get("edges", []))

    # Find stochastic node IDs
    stochastic_ids = set()
    kept_nodes = []
    for node in nodes:
        if node.get("primitive_id") in _STOCHASTIC_PRIMITIVE_IDS:
            stochastic_ids.add(node["node_id"])
        else:
            kept_nodes.append(node)

    # Remove edges referencing stochastic nodes
    kept_edges = [
        e for e in edges
        if e["source"] not in stochastic_ids and e["target"] not in stochastic_ids
    ]

    template["nodes"] = kept_nodes
    template["edges"] = kept_edges
    return template


def _try_build_graph_operator(
    modality: str, dims: Tuple[int, ...]
) -> Optional[BaseOperator]:
    """Try to build an operator from graph templates (SC-9 graph-first path).

    Looks up a graph template matching the modality, strips stochastic nodes,
    compiles via GraphCompiler, and wraps in GraphOperatorAdapter.

    Returns None on any failure (missing template, compilation error, etc.),
    allowing the caller to fall back to the modality-specific operator.
    """
    try:
        from pwm_core.graph.adapter import GraphOperatorAdapter
        from pwm_core.graph.compiler import GraphCompiler
        from pwm_core.graph.graph_spec import OperatorGraphSpec

        templates = _load_graph_templates()
        if not templates:
            return None

        # Find first template where metadata.modality matches
        matched_template = None
        for _tmpl_id, tmpl in templates.items():
            tmpl_modality = (tmpl.get("metadata") or {}).get("modality", "")
            if tmpl_modality.lower() == modality:
                matched_template = dict(tmpl)
                matched_template.setdefault("graph_id", _tmpl_id)
                break

        if matched_template is None:
            return None

        # Strip stochastic nodes (noise handled by simulator.py)
        stripped = _strip_stochastic_nodes(matched_template)

        # Remove extra fields not in OperatorGraphSpec (strict model)
        _SPEC_FIELDS = {"graph_id", "nodes", "edges", "noise_model", "metadata"}
        stripped = {k: v for k, v in stripped.items() if k in _SPEC_FIELDS}

        # Override x_shape/y_shape from dims if provided and non-default
        if dims != (64, 64):
            meta = dict(stripped.get("metadata", {}))
            meta["x_shape"] = list(dims)
            stripped["metadata"] = meta

        # Parse into OperatorGraphSpec
        spec = OperatorGraphSpec.model_validate(stripped)

        # Compile
        compiler = GraphCompiler()
        x_shape = tuple(stripped.get("metadata", {}).get("x_shape", [64, 64]))
        y_shape = tuple(stripped.get("metadata", {}).get("y_shape", list(x_shape)))
        graph_op = compiler.compile(spec, x_shape=x_shape, y_shape=y_shape)

        # Wrap in adapter
        adapter = GraphOperatorAdapter(graph_op, modality=modality)
        logger.info(
            "Built graph operator for '%s' from template '%s' (%d nodes)",
            modality, spec.graph_id, len(graph_op.forward_plan),
        )
        return adapter

    except Exception as exc:
        logger.debug(
            "Graph-first build failed for '%s': %s; falling back to modality-specific.",
            modality, exc,
        )
        return None


def build_operator(spec: ExperimentSpec) -> BaseOperator:
    """Build a physics operator from ExperimentSpec.

    Tries the graph-first path (SC-9) first: load graph template, strip
    stochastic nodes, compile via GraphCompiler, wrap in GraphOperatorAdapter.
    Falls back to modality-specific operator on failure.

    Args:
        spec: ExperimentSpec containing physics modality and operator config.

    Returns:
        A PhysicsOperator instance appropriate for the modality.
    """
    modality = spec.states.physics.modality.lower()
    dims = _get_dims_from_spec(spec)

    # Try graph-first path (SC-9)
    graph_op = _try_build_graph_operator(modality, dims)
    if graph_op is not None:
        return graph_op

    # Check if operator is explicitly specified in input
    if spec.input.operator is not None:
        op_input = spec.input.operator

        if op_input.kind == OperatorKind.matrix and op_input.matrix is not None:
            from pwm_core.physics.adapters.matrix_operator import MatrixOperator
            # Load matrix from source
            source = op_input.matrix.source
            A = np.load(source)
            return MatrixOperator(operator_id="matrix", theta={}, A=A)

        elif op_input.kind == OperatorKind.callable and op_input.callable is not None:
            from pwm_core.physics.adapters.callable_operator import CallableOperator
            import importlib
            mod = importlib.import_module(op_input.callable.module)
            fwd = getattr(mod, op_input.callable.symbol_forward)
            adj = getattr(mod, op_input.callable.symbol_adjoint)
            return CallableOperator(operator_id="callable", theta={}, fwd=fwd, adj=adj)

        elif op_input.kind == OperatorKind.parametric and op_input.parametric is not None:
            # Route based on operator_id
            operator_id = op_input.parametric.operator_id.lower()
            theta = op_input.parametric.theta_init or {}

            return _build_operator_by_id(operator_id, dims, theta, op_input.parametric.assets)

    # Route by modality name
    return _build_operator_by_id(modality, dims, {}, None)


def _build_operator_by_id(
    operator_id: str,
    dims: Tuple[int, ...],
    theta: Dict[str, Any],
    assets: Optional[Dict[str, Any]]
) -> BaseOperator:
    """Build operator by ID string."""

    operator_id = operator_id.lower()

    # Widefield / blur
    if operator_id in ("widefield", "blur", "generic"):
        return _build_widefield_operator(dims, theta)

    # Confocal (sharper PSF than widefield)
    elif operator_id == "confocal":
        theta_confocal = {"sigma": theta.get("sigma", 1.5), "mode": "reflect"}
        return _build_widefield_operator(dims, theta_confocal)

    # Structured Illumination Microscopy
    elif operator_id == "sim":
        return _build_sim_operator(dims, theta)

    # CASSI (coded aperture spectral imaging)
    elif operator_id == "cassi":
        return _build_cassi_operator(dims, theta, assets)

    # Single-Pixel Camera
    elif operator_id == "spc":
        return _build_spc_operator(dims, theta)

    # CACTI (Coded Aperture Compressive Temporal Imaging) / Video SCI
    elif operator_id in ("cacti", "sci", "video_sci", "snapshot_compressive"):
        return _build_cacti_operator(dims, theta)

    # Lensless / Diffuser camera
    elif operator_id in ("lensless", "diffuser"):
        return _build_lensless_operator(dims, theta)

    # Light-sheet microscopy
    elif operator_id == "lightsheet":
        return _build_lightsheet_operator(dims, theta)

    # CT / Tomography
    elif operator_id in ("ct", "tomography", "radon"):
        return _build_ct_operator(dims, theta)

    # MRI
    elif operator_id == "mri":
        return _build_mri_operator(dims, theta)

    # Ptychography
    elif operator_id == "ptychography":
        return _build_ptychography_operator(dims, theta)

    # Holography
    elif operator_id == "holography":
        return _build_holography_operator(dims, theta)

    # NeRF
    elif operator_id == "nerf":
        return _build_nerf_operator(dims, theta)

    # Gaussian Splatting
    elif operator_id == "gaussian_splatting":
        return _build_gaussian_splatting_operator(dims, theta)

    # OCT (Optical Coherence Tomography)
    elif operator_id == "oct":
        return _build_oct_operator(dims, theta)

    # Light Field
    elif operator_id == "light_field":
        return _build_light_field_operator(dims, theta)

    # Matrix operator
    elif operator_id == "matrix":
        return _build_widefield_operator(dims, {})

    # Identity
    elif operator_id == "identity":
        return IdentityOperator(x_shape=dims)

    # Default fallback: widefield
    return _build_widefield_operator(dims, {})


def _build_widefield_operator(dims: Tuple[int, ...], theta: Dict[str, Any]) -> BaseOperator:
    """Build a WidefieldOperator with Gaussian blur."""
    from pwm_core.physics.microscopy.widefield import WidefieldOperator

    sigma = theta.get("sigma", 2.0)
    mode = theta.get("mode", "reflect")

    return WidefieldOperator(
        operator_id="widefield",
        theta={"sigma": sigma, "mode": mode},
        x_shape=dims[:2] if len(dims) >= 2 else (64, 64),
    )


def _build_sim_operator(dims: Tuple[int, ...], theta: Dict[str, Any]) -> BaseOperator:
    """Build a SIM operator."""
    from pwm_core.physics.microscopy.sim_operator import SIMOperator

    return SIMOperator(
        operator_id="sim",
        theta=theta,
        x_shape=dims[:2] if len(dims) >= 2 else (64, 64),
        n_angles=theta.get("n_angles", 3),
        n_phases=theta.get("n_phases", 3),
        pattern_freq=theta.get("pattern_freq", 0.1),
        psf_sigma=theta.get("psf_sigma", 1.5),
    )


def _build_lightsheet_operator(dims: Tuple[int, ...], theta: Dict[str, Any]) -> BaseOperator:
    """Build a Light-sheet operator."""
    from pwm_core.physics.microscopy.lightsheet_operator import LightsheetOperator

    # Ensure 3D dims
    if len(dims) == 2:
        x_shape = (dims[0], dims[1], 32)
    else:
        x_shape = dims[:3]

    return LightsheetOperator(
        operator_id="lightsheet",
        theta=theta,
        x_shape=x_shape,
        psf_sigma=theta.get("psf_sigma", (1.5, 1.5, 1.0)),
    )


def _build_cassi_operator(
    dims: Tuple[int, ...],
    theta: Dict[str, Any],
    assets: Optional[Dict[str, Any]]
) -> BaseOperator:
    """Build a CASSIOperator for coded aperture spectral imaging."""
    from pwm_core.physics.spectral.cassi_operator import CASSIOperator

    # Determine spatial and spectral dimensions
    if len(dims) == 3:
        H, W, L = dims
    elif len(dims) == 2:
        H, W = dims
        L = theta.get("L", 8)
    else:
        H, W, L = 64, 64, 8

    # Create or load mask
    mask = None
    if assets is not None:
        mask_source = assets.get("mask")
        if mask_source is not None:
            if isinstance(mask_source, str):
                mask = np.load(mask_source)
            elif isinstance(mask_source, np.ndarray):
                mask = mask_source

    if mask is None:
        # Generate random binary coded aperture mask
        rng = np.random.default_rng(42)
        mask = (rng.random((H, W)) > 0.5).astype(np.float32)

    theta_full = {"L": L, **theta}

    return CASSIOperator(operator_id="cassi", theta=theta_full, mask=mask)


def _build_spc_operator(dims: Tuple[int, ...], theta: Dict[str, Any]) -> BaseOperator:
    """Build a Single-Pixel Camera operator."""
    from pwm_core.physics.compressive.spc_operator import SPCOperator

    x_shape = dims[:2] if len(dims) >= 2 else (64, 64)
    sampling_rate = theta.get("sampling_rate", 0.15)

    return SPCOperator(
        operator_id="spc",
        theta=theta,
        x_shape=x_shape,
        sampling_rate=sampling_rate,
    )


def _build_cacti_operator(dims: Tuple[int, ...], theta: Dict[str, Any]) -> BaseOperator:
    """Build a CACTI (video snapshot compressive imaging) operator."""
    from pwm_core.physics.compressive.cacti_operator import CACTIOperator

    # Ensure 3D dims (H, W, T)
    if len(dims) == 2:
        x_shape = (dims[0], dims[1], 8)  # Default 8 frames
    else:
        x_shape = dims[:3]

    return CACTIOperator(
        operator_id="cacti",
        theta=theta,
        x_shape=x_shape,
        shift_type=theta.get("shift_type", "vertical"),
    )


def _build_lensless_operator(dims: Tuple[int, ...], theta: Dict[str, Any]) -> BaseOperator:
    """Build a Lensless (diffuser) operator."""
    from pwm_core.physics.lensless.lensless_operator import LenslessOperator

    x_shape = dims[:2] if len(dims) >= 2 else (64, 64)

    return LenslessOperator(
        operator_id="lensless",
        theta=theta,
        x_shape=x_shape,
        psf_sigma=theta.get("psf_sigma", 10.0),
    )


def _build_ct_operator(dims: Tuple[int, ...], theta: Dict[str, Any]) -> BaseOperator:
    """Build a CT (Radon) operator."""
    from pwm_core.physics.tomography.ct_operator import CTOperator

    x_shape = dims[:2] if len(dims) >= 2 else (64, 64)

    return CTOperator(
        operator_id="ct",
        theta=theta,
        x_shape=x_shape,
        n_angles=theta.get("n_angles", 180),
    )


def _build_mri_operator(dims: Tuple[int, ...], theta: Dict[str, Any]) -> BaseOperator:
    """Build an MRI operator."""
    from pwm_core.physics.mri.mri_operator import MRIOperator

    x_shape = dims[:2] if len(dims) >= 2 else (64, 64)

    return MRIOperator(
        operator_id="mri",
        theta=theta,
        x_shape=x_shape,
        sampling_rate=theta.get("sampling_rate", 0.25),
    )


def _build_ptychography_operator(dims: Tuple[int, ...], theta: Dict[str, Any]) -> BaseOperator:
    """Build a Ptychography operator."""
    from pwm_core.physics.microscopy.ptychography_operator import PtychographyOperator

    x_shape = dims[:2] if len(dims) >= 2 else (64, 64)

    return PtychographyOperator(
        operator_id="ptychography",
        theta=theta,
        x_shape=x_shape,
        n_positions=theta.get("n_positions", 16),
        probe_size=theta.get("probe_size", 32),
    )


def _build_holography_operator(dims: Tuple[int, ...], theta: Dict[str, Any]) -> BaseOperator:
    """Build a Holography operator."""
    from pwm_core.physics.microscopy.holography_operator import HolographyOperator

    x_shape = dims[:2] if len(dims) >= 2 else (64, 64)

    return HolographyOperator(
        operator_id="holography",
        theta=theta,
        x_shape=x_shape,
        carrier_freq=theta.get("carrier_freq", 0.2),
    )


def _build_nerf_operator(dims: Tuple[int, ...], theta: Dict[str, Any]) -> BaseOperator:
    """Build a NeRF operator."""
    from pwm_core.physics.rendering.nerf_operator import NeRFOperator

    # Ensure 3D dims
    if len(dims) == 2:
        x_shape = (dims[0], dims[1], 32)
    else:
        x_shape = dims[:3]

    return NeRFOperator(
        operator_id="nerf",
        theta=theta,
        x_shape=x_shape,
        n_views=theta.get("n_views", 10),
    )


def _build_gaussian_splatting_operator(dims: Tuple[int, ...], theta: Dict[str, Any]) -> BaseOperator:
    """Build a Gaussian Splatting operator."""
    from pwm_core.physics.rendering.gaussian_splatting_operator import GaussianSplattingOperator

    # Ensure 3D dims
    if len(dims) == 2:
        x_shape = (dims[0], dims[1], 32)
    else:
        x_shape = dims[:3]

    return GaussianSplattingOperator(
        operator_id="gaussian_splatting",
        theta=theta,
        x_shape=x_shape,
        n_views=theta.get("n_views", 10),
        splat_sigma=theta.get("splat_sigma", 2.0),
    )


def _build_oct_operator(dims: Tuple[int, ...], theta: Dict[str, Any]) -> BaseOperator:
    """Build an OCT (Optical Coherence Tomography) operator."""
    from pwm_core.physics.oct.oct_operator import OCTOperator

    n_alines = theta.get("n_alines", dims[0] if len(dims) >= 1 else 128)
    n_depth = theta.get("n_depth", dims[1] if len(dims) >= 2 else 256)
    n_spectral = theta.get("n_spectral", n_depth * 2)
    dispersion_coeffs = theta.get("dispersion_coeffs", None)

    return OCTOperator(
        operator_id="oct",
        theta=theta,
        n_alines=n_alines,
        n_depth=n_depth,
        n_spectral=n_spectral,
        dispersion_coeffs=dispersion_coeffs,
    )


def _build_light_field_operator(dims: Tuple[int, ...], theta: Dict[str, Any]) -> BaseOperator:
    """Build a Light Field operator."""
    from pwm_core.physics.light_field.lf_operator import LightFieldOperator

    sx = dims[0] if len(dims) >= 1 else 64
    sy = dims[1] if len(dims) >= 2 else 64
    nu = theta.get("nu", 5)
    nv = theta.get("nv", 5)
    disparity = theta.get("disparity", 0.5)

    return LightFieldOperator(
        operator_id="light_field",
        theta=theta,
        sx=sx,
        sy=sy,
        nu=nu,
        nv=nv,
        disparity=disparity,
    )
