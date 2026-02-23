"""pwm_core.targeting.mismatch_sampler
=======================================

Sample mismatch parameters from ``contrib/mismatch_db.yaml`` and inject
them into a compiled GraphOperator to create H_nom.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_CONTRIB_DIR = Path(__file__).resolve().parents[2] / "contrib"


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file, trying yaml first."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required: pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f)


def load_mismatch_db(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the mismatch parameter database."""
    if path is None:
        path = _CONTRIB_DIR / "mismatch_db.yaml"
    data = _load_yaml(path)
    return data.get("modalities", data)


def sample_mismatch(
    modality: str,
    severity: str = "moderate",
    rng: Optional[np.random.Generator] = None,
    mismatch_db: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Sample mismatch parameters for a modality.

    Parameters
    ----------
    modality : str
        Modality name (e.g., 'cassi', 'widefield').
    severity : str
        One of 'mild', 'moderate', 'severe', 'catastrophic'.
        Controls the range of sampled parameters.
    rng : numpy Generator, optional
        Random number generator for reproducibility.
    mismatch_db : dict, optional
        Pre-loaded mismatch database. If None, loads from contrib/.

    Returns
    -------
    dict
        Parameter name -> sampled mismatch value.
    """
    if rng is None:
        rng = np.random.default_rng()

    if mismatch_db is None:
        mismatch_db = load_mismatch_db()

    if modality not in mismatch_db:
        logger.warning(
            f"Modality '{modality}' not in mismatch_db. "
            f"Available: {sorted(mismatch_db.keys())}. "
            f"Returning empty mismatch."
        )
        return {}

    mod_entry = mismatch_db[modality]
    params = mod_entry.get("parameters", {})

    severity_scale = {
        "mild": 0.25,
        "moderate": 0.50,
        "severe": 1.0,
        "catastrophic": 2.0,
    }
    scale = severity_scale.get(severity, 0.5)

    sampled: Dict[str, float] = {}
    for pname, pspec in params.items():
        prange = pspec.get("range", [0, 0])
        low, high = float(prange[0]), float(prange[1])
        center = (low + high) / 2.0
        half_range = (high - low) / 2.0

        # Scale range by severity
        scaled_low = center - half_range * scale
        scaled_high = center + half_range * scale

        sampled[pname] = float(rng.uniform(scaled_low, scaled_high))

    # Apply graph_param_map: translate conceptual param names to the actual
    # GraphOperator node parameter names for direct injection.
    # Only rename entries present in the map; keep others under original names
    # (they are retained for mismatch_magnitude computation but won't be
    # injected into the graph unless they happen to match a node param).
    graph_param_map = mod_entry.get("graph_param_map", {})
    if graph_param_map:
        remapped: Dict[str, float] = {}
        for pname, pval in sampled.items():
            graph_name = graph_param_map.get(pname, pname)
            # If multiple conceptual params map to the same graph param,
            # use the last one (highest severity weight wins via iteration order).
            remapped[graph_name] = pval
        sampled = remapped

    return sampled


def _try_rebuild_primitive(prim: Any, orig_output_size: Optional[int] = None) -> Any:
    """Attempt to rebuild a primitive from its updated ``_params``.

    Primitives that pre-compile internal matrices (e.g. RandomMask, kspace
    SubsampledFourier) must be fully reconstructed when parameters change.
    Returns the new primitive only if the output shape is unchanged; returns
    the original otherwise (prevents y-shape mismatches in the harness).
    """
    try:
        new_prim = type(prim)(dict(prim._params))
        # Shape-safety check: refuse rebuild if output dimensions changed
        if orig_output_size is not None:
            # Test with a dummy input to verify output size stability
            import numpy as _np
            dummy_in_size = getattr(prim, '_H', 32) * getattr(prim, '_W', 32)
            dummy = _np.zeros(dummy_in_size)
            try:
                new_out = new_prim.forward(dummy)
                if new_out.size != orig_output_size:
                    return prim  # Shape changed — don't rebuild
            except Exception:
                return prim
        return new_prim
    except Exception:
        return prim


def inject_mismatch(
    H_true: Any,
    theta_mismatch: Dict[str, float],
) -> Any:
    """Create H_nom by applying mismatch to H_true's parameters.

    Parameters
    ----------
    H_true : GraphOperator or GraphOperatorAdapter
        The true (ideal) operator.
    theta_mismatch : dict
        Parameter name -> mismatch value to apply.

    Returns
    -------
    H_nom
        A copy of H_true with mismatched parameters (the nominal operator).
    """
    H_nom = copy.deepcopy(H_true)

    # Try GraphOperatorAdapter's set_theta interface
    if hasattr(H_nom, "set_theta") and hasattr(H_nom, "get_theta"):
        theta = H_nom.get_theta()
        for key, val in theta_mismatch.items():
            # Try direct match
            if key in theta:
                theta[key] = val
            else:
                # Try node_id.param_name pattern
                for theta_key in theta:
                    if theta_key.endswith(f".{key}"):
                        theta[theta_key] = val
                        break
        H_nom.set_theta(theta)
    elif hasattr(H_nom, "node_map"):
        # Direct GraphOperator: inject into node params.
        # Supports two key formats produced by graph_param_map remapping:
        #   "node_id.param_name"  → route directly to the named node
        #   "param_name"          → scan all nodes for a matching param key
        for pname, pval in theta_mismatch.items():
            if "." in pname:
                # node_id.param_name format: target a specific node
                target_nid, param_name = pname.split(".", 1)
                if target_nid in H_nom.node_map:
                    nodes_to_update = [(target_nid, H_nom.node_map[target_nid])]
                else:
                    nodes_to_update = []
            else:
                # Legacy flat format: scan all nodes for a matching param key
                param_name = pname
                nodes_to_update = [
                    (nid, prim)
                    for nid, prim in H_nom.node_map.items()
                    if pname in prim._params
                ]

            for node_id, prim in nodes_to_update:
                # Measure current output size before changing params
                orig_out_size: Optional[int] = None
                try:
                    import numpy as _np
                    in_size = getattr(prim, '_H', 32) * getattr(prim, '_W', 32)
                    dummy = _np.zeros(in_size)
                    orig_out_size = prim.forward(dummy).size
                except Exception:
                    pass

                # Coerce integer-typed params (e.g. seed, n_angles) so that
                # primitives using np.random.default_rng(seed) don't fail.
                existing_type = type(prim._params.get(param_name, pval))
                if existing_type is int or param_name in ("seed", "n_angles", "T"):
                    pval = int(round(float(pval)))
                prim._params[param_name] = pval

                # Some primitives (e.g. RandomMask, SubsampledFourier, CTRadon)
                # pre-compile their measurement matrices at __init__ and don't
                # re-read _params on forward().  Rebuild the primitive so the
                # new param takes effect, but only if the output shape is
                # preserved (prevents y-shape breaks in the harness).
                new_prim = _try_rebuild_primitive(prim, orig_out_size)
                if new_prim is not prim:
                    H_nom.node_map[node_id] = new_prim
                    for i, (nid, p) in enumerate(H_nom.forward_plan):
                        if nid == node_id:
                            H_nom.forward_plan[i] = (nid, new_prim)
                    for i, (nid, p) in enumerate(H_nom.adjoint_plan):
                        if nid == node_id:
                            H_nom.adjoint_plan[i] = (nid, new_prim)
    else:
        logger.warning(
            "Cannot inject mismatch: operator has no set_theta() or node_map"
        )

    return H_nom


def compute_mismatch_magnitude(theta_mismatch: Dict[str, float]) -> float:
    """Compute the L2 magnitude of the mismatch vector."""
    if not theta_mismatch:
        return 0.0
    values = np.array(list(theta_mismatch.values()), dtype=np.float64)
    return float(np.linalg.norm(values))
