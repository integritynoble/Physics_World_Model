"""Agent JSON → OperatorGraphSpec translator.

Bridges the gap between the free-form FlowchartElement schema produced by
the Plan Agent and the formally typed OperatorGraphSpec required by the
Constrained Primitive Compiler.

The translator maps each FlowchartElement to one or more canonical
primitives using deterministic rules, ensuring the output is always a
valid composition over the 11-primitive basis.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from pwm_core.graph.graph_spec import (
    GraphEdge,
    GraphNode,
    NoiseSpec,
    OperatorGraphSpec,
)
from pwm_core.graph.ir_types import (
    CanonicalPrimitive,
    DetectFamily,
    NodeRole,
    TransformFamily,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mapping tables: FlowchartElement type → canonical primitive(s)
# ---------------------------------------------------------------------------

# Map (element_type, model_hint) → (primitive_id, canonical_id, node_role)
# model_hint is derived from element parameters (e.g. "beer_lambert", "radon")

_SOURCE_PRIMITIVES = {
    "xray": ("xray_source", None, NodeRole.source),
    "photon": ("photon_source", None, NodeRole.source),
    "acoustic": ("acoustic_source", None, NodeRole.source),
    "spin": ("spin_source", None, NodeRole.source),
    "electron": ("electron_beam_source", None, NodeRole.source),
    "default": ("generic_source", None, NodeRole.source),
}

_INTERACTION_PRIMITIVES = {
    "beer_lambert": ("beer_lambert", CanonicalPrimitive.Lambda, NodeRole.interaction),
    "absorption": ("optical_absorption", CanonicalPrimitive.M, NodeRole.interaction),
    "attenuation": ("beer_lambert", CanonicalPrimitive.Lambda, NodeRole.interaction),
    "scatter": ("scatter_model", CanonicalPrimitive.R, NodeRole.interaction),
    "compton": ("scatter_model", CanonicalPrimitive.R, NodeRole.interaction),
    "raman": ("scatter_model", CanonicalPrimitive.R, NodeRole.interaction),
    "fluorescence": ("fluorescence_kinetics", CanonicalPrimitive.R, NodeRole.interaction),
    "relaxation": ("coded_mask", CanonicalPrimitive.M, NodeRole.interaction),
    "tissue": ("coded_mask", CanonicalPrimitive.M, NodeRole.interaction),
    "modulation": ("coded_mask", CanonicalPrimitive.M, NodeRole.interaction),
    "default": ("coded_mask", CanonicalPrimitive.M, NodeRole.interaction),
}

_GEOMETRY_PRIMITIVES = {
    "radon": ("ct_radon", CanonicalPrimitive.Pi, NodeRole.transport),
    "parallel_beam": ("ct_radon", CanonicalPrimitive.Pi, NodeRole.transport),
    "fan_beam": ("ct_radon", CanonicalPrimitive.Pi, NodeRole.transport),
    "cone_beam": ("cone_beam_radon", CanonicalPrimitive.Pi, NodeRole.transport),
    "kspace": ("mri_kspace", CanonicalPrimitive.F, NodeRole.transport),
    "cartesian": ("mri_kspace", CanonicalPrimitive.F, NodeRole.transport),
    "radial": ("mri_kspace", CanonicalPrimitive.F, NodeRole.transport),
    "spiral": ("mri_kspace", CanonicalPrimitive.F, NodeRole.transport),
    "fresnel": ("fresnel_prop", CanonicalPrimitive.P, NodeRole.transport),
    "propagation": ("fresnel_prop", CanonicalPrimitive.P, NodeRole.transport),
    "psf": ("conv2d", CanonicalPrimitive.C, NodeRole.transport),
    "convolution": ("conv2d", CanonicalPrimitive.C, NodeRole.transport),
    "airy": ("conv2d", CanonicalPrimitive.C, NodeRole.transport),
    "objective": ("conv2d", CanonicalPrimitive.C, NodeRole.transport),
    "dispersion": ("spectral_dispersion", CanonicalPrimitive.W, NodeRole.transport),
    "coded_aperture": ("coded_mask", CanonicalPrimitive.M, NodeRole.transport),
    "random_mask": ("random_mask", CanonicalPrimitive.S, NodeRole.transport),
    "undersampling": ("random_mask", CanonicalPrimitive.S, NodeRole.transport),
    "default": ("identity", CanonicalPrimitive.M, NodeRole.transport),
}

_DETECTOR_PRIMITIVES = {
    "intensity": ("magnitude_sq", CanonicalPrimitive.D, NodeRole.sensor),
    "photon_counting": ("magnitude_sq", CanonicalPrimitive.D, NodeRole.sensor),
    "ccd": ("magnitude_sq", CanonicalPrimitive.D, NodeRole.sensor),
    "cmos": ("magnitude_sq", CanonicalPrimitive.D, NodeRole.sensor),
    "interferometric": ("magnitude_sq", CanonicalPrimitive.D, NodeRole.sensor),
    "default": ("generic_sensor", CanonicalPrimitive.D, NodeRole.sensor),
}


# ---------------------------------------------------------------------------
# Hint extraction
# ---------------------------------------------------------------------------


def _extract_hint(element: Dict[str, Any]) -> str:
    """Extract a model hint from FlowchartElement parameters."""
    params = element.get("parameters", {})
    name = element.get("name", "").lower()
    eid = element.get("id", "").lower()

    # Check explicit model key
    model = str(params.get("model", "")).lower()
    if model:
        return model

    # Check scan_type
    scan_type = str(params.get("scan_type", "")).lower()
    if scan_type:
        return scan_type

    # Check known keywords in name
    keywords = [
        "radon", "kspace", "k-space", "fresnel", "psf", "airy",
        "beer_lambert", "beer-lambert", "attenuation", "scatter",
        "compton", "raman", "fluorescen", "dispersion", "coded",
        "fan_beam", "fan-beam", "cone_beam", "cone-beam",
        "parallel", "cartesian", "radial", "spiral",
        "intensity", "photon_counting", "ccd", "cmos",
        "interferometric", "homodyne", "heterodyne",
        "propagation", "objective", "convolution",
        "absorption", "relaxation", "tissue", "modulation",
        "undersampling", "random",
        "xray", "x-ray", "photon", "acoustic", "spin", "electron",
    ]
    combined = f"{name} {eid} {model}"
    for kw in keywords:
        if kw in combined:
            # Normalise dashes to underscores
            return kw.replace("-", "_").replace(" ", "_")

    return "default"


def _detect_carrier(elements: List[Dict[str, Any]]) -> str:
    """Detect the physical carrier from source element parameters."""
    for el in elements:
        if el.get("type") == "source":
            name = el.get("name", "").lower()
            eid = el.get("id", "").lower()
            combined = f"{name} {eid}"
            if any(k in combined for k in ["xray", "x-ray", "x_ray"]):
                return "xray"
            if any(k in combined for k in ["acoustic", "ultrasound", "transducer"]):
                return "acoustic"
            if any(k in combined for k in ["spin", "magnet", "mri", "b0"]):
                return "spin"
            if any(k in combined for k in ["electron", "beam"]):
                return "electron"
    return "photon"


# ---------------------------------------------------------------------------
# AgentToGraphTranslator
# ---------------------------------------------------------------------------


class AgentToGraphTranslator:
    """Convert agent-produced forward model specs to typed OperatorGraphSpecs.

    The translator maps each FlowchartElement to one or more GraphNodes
    whose ``canonical_id`` is one of the 11 canonical primitives.  This
    ensures the output is always within the scope of the Finite Primitive
    Basis Theorem.
    """

    def translate(
        self,
        action: Dict[str, Any],
        modality: str = "generic",
        period: str = "forward",
    ) -> OperatorGraphSpec:
        """Convert a forward action dict to an OperatorGraphSpec.

        Parameters
        ----------
        action : dict
            The ``action`` section from the agent's spec JSON.  Must
            contain ``elements`` (list of FlowchartElement dicts) and
            optionally ``measurement_shape``.
        modality : str
            Modality identifier (e.g. "ct", "mri").
        period : str
            "forward" or "reconstruction".

        Returns
        -------
        OperatorGraphSpec
            Formally typed graph ready for the Constrained Primitive Compiler.
        """
        elements = action.get("elements", [])
        if not elements:
            raise ValueError("Action contains no elements — cannot translate.")

        carrier = _detect_carrier(elements)
        nodes: List[GraphNode] = []
        edges: List[GraphEdge] = []
        node_id_map: Dict[str, str] = {}  # element_id -> graph_node_id

        # --- 1. Translate each element ---
        for el in elements:
            el_type = el.get("type", "other")
            el_id = el.get("id", f"node_{len(nodes)}")
            hint = _extract_hint(el)
            params = dict(el.get("parameters", {}))

            prim_id, canonical_id, role = self._resolve_primitive(
                el_type, hint, carrier, params
            )

            # Create graph node
            graph_node_id = f"{el_id}"
            node = GraphNode(
                node_id=graph_node_id,
                primitive_id=prim_id,
                params=self._sanitize_params(prim_id, params),
                role=role,
                canonical_id=canonical_id,
            )
            nodes.append(node)
            node_id_map[el_id] = graph_node_id

        # --- 2. Build edges from connects_to ---
        for el in elements:
            el_id = el.get("id", "")
            src_id = node_id_map.get(el_id)
            if not src_id:
                continue
            for target_el_id in el.get("connects_to", []):
                tgt_id = node_id_map.get(target_el_id)
                if tgt_id:
                    edges.append(GraphEdge(source=src_id, target=tgt_id))

        # --- 3. If no connects_to, create sequential chain ---
        if not edges and len(nodes) > 1:
            for i in range(len(nodes) - 1):
                edges.append(GraphEdge(
                    source=nodes[i].node_id,
                    target=nodes[i + 1].node_id,
                ))

        # --- 4. Ensure a noise terminal node ---
        has_noise = any(n.role == NodeRole.noise for n in nodes)
        if not has_noise:
            noise_id = "sensor_noise"
            noise_type = self._detect_noise_type(elements)
            noise_params = self._extract_noise_params(elements)
            nodes.append(GraphNode(
                node_id=noise_id,
                primitive_id=noise_type,
                params=noise_params,
                role=NodeRole.noise,
            ))
            # Connect last non-noise node to noise
            if nodes:
                last_non_noise = [n for n in nodes if n.role != NodeRole.noise][-1]
                edges.append(GraphEdge(
                    source=last_non_noise.node_id,
                    target=noise_id,
                ))

        # --- 5. Parse measurement shape ---
        mshape_str = action.get("measurement_shape", "")
        y_shape = self._parse_shape(mshape_str) or [64, 64]

        # --- 6. Assemble OperatorGraphSpec ---
        spec = OperatorGraphSpec(
            graph_id=f"{modality}_{period}_agent",
            nodes=nodes,
            edges=edges,
            noise_model=NoiseSpec(
                noise_type="poisson_gaussian",
                params=self._extract_noise_params(elements),
            ),
            metadata={
                "modality": modality,
                "period": period,
                "carrier": carrier,
                "agent_generated": True,
                "x_shape": [64, 64],
                "y_shape": y_shape,
            },
        )

        logger.info(
            f"Translated agent spec → {spec.graph_id}: "
            f"{len(nodes)} nodes, {len(edges)} edges"
        )
        return spec

    # -----------------------------------------------------------------------
    # Resolution logic
    # -----------------------------------------------------------------------

    def _resolve_primitive(
        self,
        el_type: str,
        hint: str,
        carrier: str,
        params: Dict[str, Any],
    ) -> Tuple[str, Optional[CanonicalPrimitive], Optional[NodeRole]]:
        """Map (element_type, hint) to (primitive_id, canonical_id, role)."""

        if el_type == "source":
            key = carrier if carrier in _SOURCE_PRIMITIVES else "default"
            return _SOURCE_PRIMITIVES[key]

        if el_type == "interaction":
            for key in [hint] + [hint.split("_")[0]] + ["default"]:
                if key in _INTERACTION_PRIMITIVES:
                    return _INTERACTION_PRIMITIVES[key]
            return _INTERACTION_PRIMITIVES["default"]

        if el_type == "geometry":
            for key in [hint] + [hint.split("_")[0]] + ["default"]:
                if key in _GEOMETRY_PRIMITIVES:
                    return _GEOMETRY_PRIMITIVES[key]
            return _GEOMETRY_PRIMITIVES["default"]

        if el_type == "detector":
            for key in [hint] + [hint.split("_")[0]] + ["default"]:
                if key in _DETECTOR_PRIMITIVES:
                    return _DETECTOR_PRIMITIVES[key]
            return _DETECTOR_PRIMITIVES["default"]

        if el_type == "digitization":
            # ADC / quantization → treated as part of detection
            return ("adc_clip", CanonicalPrimitive.D, NodeRole.readout)

        if el_type == "processing":
            # Processing steps (tube lens, relay) → propagation or convolution
            for key in [hint] + ["default"]:
                if key in _GEOMETRY_PRIMITIVES:
                    return _GEOMETRY_PRIMITIVES[key]
            return ("identity", None, NodeRole.utility)

        # Fallback
        return ("identity", None, NodeRole.utility)

    def _sanitize_params(
        self, primitive_id: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Clean agent params for the target primitive.

        Removes keys that are not understood by the primitive and adds
        sensible defaults where needed.
        """
        # Only pass through numeric/string params, skip complex objects
        clean = {}
        for k, v in params.items():
            if isinstance(v, (int, float, str, bool)):
                clean[k] = v
            elif isinstance(v, list) and all(isinstance(i, (int, float)) for i in v):
                clean[k] = v
        return clean

    def _detect_noise_type(self, elements: List[Dict[str, Any]]) -> str:
        """Detect the appropriate noise primitive from element noise specs."""
        has_poisson = False
        has_gaussian = False
        for el in elements:
            for noise in el.get("noise", []):
                ntype = noise.get("type", "").lower()
                if "poisson" in ntype:
                    has_poisson = True
                if "gaussian" in ntype or "read" in ntype:
                    has_gaussian = True
        if has_poisson and has_gaussian:
            return "poisson_gaussian_sensor"
        if has_poisson:
            return "poisson_only_sensor"
        return "poisson_gaussian_sensor"

    def _extract_noise_params(self, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract noise parameters from element noise specs."""
        params: Dict[str, Any] = {}
        for el in elements:
            for noise in el.get("noise", []):
                nparams = noise.get("parameters", {})
                ntype = noise.get("type", "").lower()
                if "poisson" in ntype:
                    if "I0" in nparams:
                        params["peak_photons"] = float(nparams["I0"])
                    elif "i0" in nparams:
                        params["peak_photons"] = float(nparams["i0"])
                if "gaussian" in ntype or "read" in ntype:
                    for k in ["sigma", "sigma_electrons", "read_noise"]:
                        if k in nparams:
                            params["read_sigma"] = float(nparams[k])
                            break
        return params

    def _parse_shape(self, shape_str: str) -> Optional[List[int]]:
        """Parse a shape string like '(180, 512)' or '(8, 256, 64)'."""
        if not shape_str:
            return None
        cleaned = shape_str.strip().strip("()")
        try:
            return [int(x.strip()) for x in cleaned.split(",") if x.strip()]
        except (ValueError, AttributeError):
            return None
