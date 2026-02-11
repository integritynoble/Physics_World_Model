"""CasePack acceptance test runner framework."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict
import numpy as np


class CasePackResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    modality: str
    forward_ok: bool = False
    recon_psnr: Optional[float] = None
    primary_metric: Optional[float] = None
    primary_metric_name: Optional[str] = None
    secondary_metric: Optional[float] = None
    secondary_metric_name: Optional[str] = None
    calib_nll_improvement: Optional[float] = None
    ident_flags: Dict[str, bool] = {}


def run_casepack(modality: str, quick: bool = True) -> CasePackResult:
    """Run acceptance tests for a single modality.

    Parameters
    ----------
    modality : str
        Modality key (e.g. 'ct', 'mri', 'cassi').
    quick : bool
        If True, use tiny data (CPU, <30s). If False, full-size (nightly).

    Returns
    -------
    CasePackResult
    """
    from pwm_core.core.modality_registry import get_modality
    from pwm_core.core.metric_registry import build_metric, PSNR

    result = CasePackResult(modality=modality)

    try:
        mod_info = get_modality(modality)
    except KeyError:
        return result

    # Determine sizes
    quick_sizes = {
        "ct": (32, 32), "mri": (32, 32), "cassi": (16, 16, 8),
        "spc": (16, 16), "cacti": (16, 16, 4), "ultrasound": (32, 32),
        "sem": (32, 32), "tem": (32, 32), "pet": (32, 32),
        "spect": (32, 32), "electron_tomography": (32, 32),
    }
    default_size = (16, 16)
    x_shape = quick_sizes.get(modality, default_size) if quick else (64, 64)

    # Try to build operator and run forward
    try:
        from pwm_core.graph.compiler import GraphCompiler
        from pwm_core.graph.graph_spec import OperatorGraphSpec
        import yaml, os

        templates_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "contrib", "graph_templates.yaml"
        )
        template_id = mod_info.default_template_id or f"{modality}_graph_v2"

        with open(templates_path) as f:
            data = yaml.safe_load(f)
        templates = data.get("templates", {})

        if template_id not in templates:
            # Try v1 fallback
            template_id = f"{modality}_graph_v1"

        if template_id in templates:
            tpl = dict(templates[template_id])
            tpl.pop("description", None)
            spec = OperatorGraphSpec.model_validate({"graph_id": template_id, **tpl})
            compiler = GraphCompiler()
            graph = compiler.compile(spec, x_shape=x_shape[:2] if len(x_shape) >= 2 else x_shape)

            # Mode S: forward sanity
            rng = np.random.RandomState(42)
            x = rng.rand(*x_shape).astype(np.float32)
            y = graph.forward(x)

            forward_ok = (
                y is not None
                and np.isfinite(y).all()
                and y.shape[0] > 0
            )
            result.forward_ok = forward_ok

            if forward_ok:
                # Mode I: basic reconstruction (adjoint as proxy)
                try:
                    x_hat = graph.adjoint(y)
                    psnr_metric = PSNR()
                    result.recon_psnr = psnr_metric(x_hat, x)
                    result.primary_metric = result.recon_psnr
                    result.primary_metric_name = "psnr"
                except Exception:
                    pass
    except Exception:
        pass

    return result
