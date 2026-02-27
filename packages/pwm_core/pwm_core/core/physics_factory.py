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
- photoacoustic: PAOperator (circular Radon transform)
- fpm: FPMOperator (Fourier ptychographic microscopy)
- flim: FLIMOperator (fluorescence lifetime imaging)
- dot: DOTOperator (diffuse optical tomography)
- integral: IntegralOperator (plenoptic / integral photography)
- phase_retrieval / cdi: CDIOperator (coherent diffraction imaging)
- ultrasound: UltrasoundOperator (pulse-echo RF model)
- cryo_em: CryoEMOperator (cryo-EM CTF + B-factor)
- cbct: CBCTOperator (cone-beam CT)
- compressive_holography: CompressiveHolographyOperator (multi-depth Fresnel)
- fluorescence_microscopy: FluorescenceMicroscopyOperator (dual-PSF Stokes)
- sem: SEMOperator (scanning electron microscopy)
- tem: TEMOperator (transmission electron microscopy / CTF)
- electron_tomography: ETOperator (tilt-series projection)
- pet: PETOperator (positron emission tomography)
- spect: SPECTOperator (single photon emission CT)
- xray_radiography: XRayRadiographyOperator (Beer-Lambert)
- matrix: MatrixOperator (explicit matrix A)
- callable: CallableOperator (user-provided forward/adjoint)
- identity: fallback for testing

Unrecognized modalities fall through to graph-first path (YAML templates),
then to widefield fallback.
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
        # Format: {"x": [H, W, L], "y": [H, W]} — use x dims
        if "x" in dims:
            x_dims = dims["x"]
            if isinstance(x_dims, (list, tuple)):
                return tuple(int(d) for d in x_dims)
        # Format: {"H": ..., "W": ..., "L": ...}
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

    # Explicit operator specification takes priority over graph-first path
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

    # Try graph-first path (SC-9) when no explicit operator specified
    graph_op = _try_build_graph_operator(modality, dims)
    if graph_op is not None:
        return graph_op

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

    # Photoacoustic
    elif operator_id == "photoacoustic":
        return _build_photoacoustic_operator(dims, theta)

    # FPM (Fourier Ptychographic Microscopy)
    elif operator_id == "fpm":
        return _build_fpm_operator(dims, theta)

    # FLIM (Fluorescence Lifetime Imaging)
    elif operator_id == "flim":
        return _build_flim_operator(dims, theta)

    # DOT (Diffuse Optical Tomography)
    elif operator_id in ("dot", "diffuse_optical"):
        return _build_dot_operator(dims, theta)

    # Integral Photography (Plenoptic)
    elif operator_id == "integral":
        return _build_integral_operator(dims, theta)

    # Phase Retrieval / CDI
    elif operator_id in ("phase_retrieval", "cdi"):
        return _build_cdi_operator(dims, theta)

    # Ultrasound
    elif operator_id == "ultrasound":
        return _build_ultrasound_operator(dims, theta)

    # SEM (Scanning Electron Microscopy)
    elif operator_id == "sem":
        return _build_sem_operator(dims, theta)

    # TEM (Transmission Electron Microscopy)
    elif operator_id == "tem":
        return _build_tem_operator(dims, theta)

    # Electron Tomography
    elif operator_id == "electron_tomography":
        return _build_et_operator(dims, theta)

    # PET (Positron Emission Tomography)
    elif operator_id == "pet":
        return _build_pet_operator(dims, theta)

    # SPECT (Single Photon Emission CT)
    elif operator_id == "spect":
        return _build_spect_operator(dims, theta)

    # X-ray Radiography
    elif operator_id == "xray_radiography":
        return _build_xray_radiography_operator(dims, theta)

    # Cryo-EM
    elif operator_id in ("cryo_em", "cryoem"):
        return _build_cryoem_operator(dims, theta)

    # CBCT (Cone-Beam CT)
    elif operator_id in ("cbct", "cone_beam_ct"):
        return _build_cbct_operator(dims, theta)

    # Compressive Holography
    elif operator_id in ("compressive_holography", "comp_holo"):
        return _build_compressive_holography_operator(dims, theta)

    # Fluorescence Microscopy
    elif operator_id in ("fluorescence_microscopy", "fluorescence"):
        return _build_fluorescence_microscopy_operator(dims, theta)

    # Default fallback: try graph-first, then widefield
    graph_op = _try_build_graph_operator(operator_id, dims)
    if graph_op is not None:
        return graph_op
    logger.warning("No operator or graph template for '%s', using widefield fallback", operator_id)
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


def _build_photoacoustic_operator(dims: Tuple[int, ...], theta: Dict[str, Any]) -> BaseOperator:
    """Build a Photoacoustic operator."""
    from pwm_core.physics.photoacoustic.pa_operator import PAOperator

    ny = dims[0] if len(dims) >= 1 else 64
    nx = dims[1] if len(dims) >= 2 else 64

    return PAOperator(
        operator_id="photoacoustic",
        theta=theta,
        ny=ny,
        nx=nx,
        n_transducers=theta.get("n_transducers", 32),
        speed_of_sound=theta.get("speed_of_sound", 1.0),
    )


def _build_fpm_operator(dims: Tuple[int, ...], theta: Dict[str, Any]) -> BaseOperator:
    """Build an FPM (Fourier Ptychographic Microscopy) operator."""
    from pwm_core.physics.fpm.fpm_operator import FPMOperator

    hr_size = dims[0] if len(dims) >= 1 else 128

    return FPMOperator(
        operator_id="fpm",
        theta=theta,
        hr_size=hr_size,
        lr_size=theta.get("lr_size", hr_size // 4),
        na=theta.get("na", 0.1),
    )


def _build_flim_operator(dims: Tuple[int, ...], theta: Dict[str, Any]) -> BaseOperator:
    """Build a FLIM (Fluorescence Lifetime Imaging) operator."""
    from pwm_core.physics.flim.flim_operator import FLIMOperator

    ny = dims[0] if len(dims) >= 1 else 32
    nx = dims[1] if len(dims) >= 2 else 32

    return FLIMOperator(
        operator_id="flim",
        theta=theta,
        ny=ny,
        nx=nx,
        n_time_bins=theta.get("n_time_bins", 64),
        time_range_ns=theta.get("time_range_ns", 12.5),
        irf_sigma_ns=theta.get("irf_sigma_ns", 0.3),
    )


def _build_dot_operator(dims: Tuple[int, ...], theta: Dict[str, Any]) -> BaseOperator:
    """Build a DOT (Diffuse Optical Tomography) operator."""
    from pwm_core.physics.diffuse_optical.dot_operator import DOTOperator

    grid_size = dims[0] if len(dims) >= 1 else 16

    return DOTOperator(
        operator_id="dot",
        theta=theta,
        n_sources=theta.get("n_sources", 8),
        n_detectors=theta.get("n_detectors", 8),
        grid_size=grid_size,
        mu_a_bg=theta.get("mu_a_bg", 0.01),
        mu_s_prime=theta.get("mu_s_prime", 1.0),
    )


def _build_integral_operator(dims: Tuple[int, ...], theta: Dict[str, Any]) -> BaseOperator:
    """Build an Integral Photography (plenoptic) operator."""
    from pwm_core.physics.integral.integral_operator import IntegralOperator

    ny = dims[0] if len(dims) >= 1 else 64
    nx = dims[1] if len(dims) >= 2 else 64

    return IntegralOperator(
        operator_id="integral",
        theta=theta,
        ny=ny,
        nx=nx,
        n_depths=theta.get("n_depths", 8),
    )


def _build_cdi_operator(dims: Tuple[int, ...], theta: Dict[str, Any]) -> BaseOperator:
    """Build a CDI (phase retrieval) operator."""
    from pwm_core.physics.phase_retrieval.cdi_operator import CDIOperator

    ny = dims[0] if len(dims) >= 1 else 64
    nx = dims[1] if len(dims) >= 2 else 64

    return CDIOperator(
        operator_id="phase_retrieval",
        theta=theta,
        ny=ny,
        nx=nx,
        oversampling=theta.get("oversampling", 2),
    )


def _build_ultrasound_operator(dims: Tuple[int, ...], theta: Dict[str, Any]) -> BaseOperator:
    """Build an Ultrasound operator."""
    from pwm_core.physics.ultrasound.ultrasound_operator import UltrasoundOperator

    nz = dims[0] if len(dims) >= 1 else 64
    nx = dims[1] if len(dims) >= 2 else 64

    return UltrasoundOperator(
        operator_id="ultrasound",
        theta=theta,
        nz=nz,
        nx=nx,
        n_elements=theta.get("n_elements", 32),
        n_samples=theta.get("n_samples", 128),
    )


def _build_sem_operator(dims: Tuple[int, ...], theta: Dict[str, Any]) -> BaseOperator:
    """Build a SEM (Scanning Electron Microscopy) operator."""
    from pwm_core.physics.electron.sem_operator import SEMOperator

    ny = dims[0] if len(dims) >= 1 else 64
    nx = dims[1] if len(dims) >= 2 else 64

    return SEMOperator(
        operator_id="sem",
        theta=theta,
        ny=ny,
        nx=nx,
        voltage_kv=theta.get("voltage_kv", 15.0),
        psf_sigma=theta.get("psf_sigma", 1.0),
    )


def _build_tem_operator(dims: Tuple[int, ...], theta: Dict[str, Any]) -> BaseOperator:
    """Build a TEM (Transmission Electron Microscopy) operator."""
    from pwm_core.physics.electron.tem_operator import TEMOperator

    ny = dims[0] if len(dims) >= 1 else 64
    nx = dims[1] if len(dims) >= 2 else 64

    return TEMOperator(
        operator_id="tem",
        theta=theta,
        ny=ny,
        nx=nx,
        defocus_nm=theta.get("defocus_nm", -50.0),
        Cs_mm=theta.get("Cs_mm", 1.0),
        wavelength_pm=theta.get("wavelength_pm", 2.51),
    )


def _build_et_operator(dims: Tuple[int, ...], theta: Dict[str, Any]) -> BaseOperator:
    """Build an Electron Tomography operator."""
    from pwm_core.physics.electron.et_operator import ETOperator

    if len(dims) == 2:
        D, H, W = 32, dims[0], dims[1]
    elif len(dims) >= 3:
        D, H, W = dims[0], dims[1], dims[2]
    else:
        D, H, W = 32, 64, 64

    return ETOperator(
        operator_id="electron_tomography",
        theta=theta,
        D=D,
        H=H,
        W=W,
        n_tilts=theta.get("n_tilts", 16),
    )


def _build_pet_operator(dims: Tuple[int, ...], theta: Dict[str, Any]) -> BaseOperator:
    """Build a PET (Positron Emission Tomography) operator."""
    from pwm_core.physics.nuclear.pet_operator import PETOperator

    ny = dims[0] if len(dims) >= 1 else 64
    nx = dims[1] if len(dims) >= 2 else 64

    return PETOperator(
        operator_id="pet",
        theta=theta,
        ny=ny,
        nx=nx,
        n_angles=theta.get("n_angles", 32),
    )


def _build_spect_operator(dims: Tuple[int, ...], theta: Dict[str, Any]) -> BaseOperator:
    """Build a SPECT operator."""
    from pwm_core.physics.nuclear.spect_operator import SPECTOperator

    ny = dims[0] if len(dims) >= 1 else 64
    nx = dims[1] if len(dims) >= 2 else 64

    return SPECTOperator(
        operator_id="spect",
        theta=theta,
        ny=ny,
        nx=nx,
        n_angles=theta.get("n_angles", 32),
        collimator_sigma=theta.get("collimator_sigma", 2.0),
    )


def _build_xray_radiography_operator(dims: Tuple[int, ...], theta: Dict[str, Any]) -> BaseOperator:
    """Build an X-ray Radiography operator."""
    from pwm_core.physics.radiography.xray_radiography_operator import XRayRadiographyOperator

    ny = dims[0] if len(dims) >= 1 else 64
    nx = dims[1] if len(dims) >= 2 else 64

    return XRayRadiographyOperator(
        operator_id="xray_radiography",
        theta=theta,
        ny=ny,
        nx=nx,
        mu=theta.get("mu", 1.0),
        psf_sigma=theta.get("psf_sigma", 0.5),
    )


def _build_cryoem_operator(dims: Tuple[int, ...], theta: Dict[str, Any]) -> BaseOperator:
    """Build a Cryo-EM operator."""
    from pwm_core.physics.electron.cryoem_operator import CryoEMOperator

    ny = dims[0] if len(dims) >= 1 else 64
    nx = dims[1] if len(dims) >= 2 else 64

    return CryoEMOperator(
        operator_id="cryo_em",
        theta=theta,
        ny=ny,
        nx=nx,
        defocus_nm=theta.get("defocus_nm", -500.0),
        Cs_mm=theta.get("Cs_mm", 2.0),
        wavelength_pm=theta.get("wavelength_pm", 2.51),
        B_factor=theta.get("B_factor", 50.0),
        ice_thickness_nm=theta.get("ice_thickness_nm", 50.0),
    )


def _build_cbct_operator(dims: Tuple[int, ...], theta: Dict[str, Any]) -> BaseOperator:
    """Build a CBCT (Cone-Beam CT) operator."""
    from pwm_core.physics.tomography.cbct_operator import CBCTOperator

    ny = dims[0] if len(dims) >= 1 else 64
    nx = dims[1] if len(dims) >= 2 else 64

    return CBCTOperator(
        operator_id="cbct",
        theta=theta,
        ny=ny,
        nx=nx,
        n_angles=theta.get("n_angles", 180),
        n_det=theta.get("n_det", int(nx * 1.44)),
        D_so=theta.get("D_so", 100.0),
        D_sd=theta.get("D_sd", 150.0),
        detector_offset=theta.get("detector_offset", 0.0),
    )


def _build_compressive_holography_operator(dims: Tuple[int, ...], theta: Dict[str, Any]) -> BaseOperator:
    """Build a Compressive Holography operator."""
    from pwm_core.physics.microscopy.compressive_holography_operator import CompressiveHolographyOperator

    ny = dims[0] if len(dims) >= 1 else 64
    nx = dims[1] if len(dims) >= 2 else 64

    return CompressiveHolographyOperator(
        operator_id="compressive_holography",
        theta=theta,
        ny=ny,
        nx=nx,
        n_depths=theta.get("n_depths", 4),
        depth_spacing_um=theta.get("depth_spacing_um", 100.0),
        wavelength_nm=theta.get("wavelength_nm", 532.0),
        carrier_freq=theta.get("carrier_freq", 0.15),
    )


def _build_fluorescence_microscopy_operator(dims: Tuple[int, ...], theta: Dict[str, Any]) -> BaseOperator:
    """Build a Fluorescence Microscopy operator."""
    from pwm_core.physics.microscopy.fluorescence_operator import FluorescenceMicroscopyOperator

    ny = dims[0] if len(dims) >= 1 else 64
    nx = dims[1] if len(dims) >= 2 else 64

    return FluorescenceMicroscopyOperator(
        operator_id="fluorescence_microscopy",
        theta=theta,
        ny=ny,
        nx=nx,
        psf_sigma_ex=theta.get("psf_sigma_ex", 1.5),
        psf_sigma_em=theta.get("psf_sigma_em", 2.0),
        quantum_yield=theta.get("quantum_yield", 0.7),
        background=theta.get("background", 0.02),
    )
