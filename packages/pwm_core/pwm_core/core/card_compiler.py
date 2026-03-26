"""Card-to-Judge compilation bridge.

Connects docking artifacts (Cards) to the trust pipeline:
  Card -> compile -> Harness.run -> RunBundle -> Judge -> Certificate

This closes the card lifecycle loop described in the strategy section 8.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def compile_spec_card(spec_card: Dict[str, Any]) -> Dict[str, Any]:
    """Compile a SpecCard into Harness-compatible parameters."""
    return {
        "modality": spec_card.get("modality", spec_card.get("domain", "")),
        "solver": spec_card.get("default_solver", "traditional_cpu"),
        "track": spec_card.get("track", "correct"),
        "budget_s": spec_card.get("budget_s", 600),
    }


def run_card(
    spec_card: Dict[str, Any],
    method_card: Optional[Dict[str, Any]] = None,
    n_scenes: int = 5,
    seed: int = 42,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a SpecCard through the full trust pipeline.

    Returns dict with: bundle_path, cert_path, trust_tier, gate_verdicts.
    """
    from pwm_core.targeting.harness import Harness
    from pwm_core.targeting.runbundle_emitter import emit_runbundle, issue_certificate

    params = compile_spec_card(spec_card)
    if method_card:
        params["solver"] = method_card.get("solver_name", params["solver"])

    harness = Harness(
        modality=params["modality"],
        solver=params["solver"],
        track=params["track"],
        budget_s=params["budget_s"],
    )

    result = harness.run(n_scenes=n_scenes, seed=seed)
    out = Path(output_dir) if output_dir else Path(".")
    bundle_path = emit_runbundle(result, out)
    cert_path = issue_certificate(bundle_path)

    import json
    cert = json.loads(cert_path.read_text()) if cert_path else {}

    return {
        "bundle_path": str(bundle_path),
        "cert_path": str(cert_path) if cert_path else None,
        "trust_tier": cert.get("trust_tier"),
        "gate_verdicts": {
            k: v.get("verdict") if isinstance(v, dict) else v
            for k, v in cert.get("gate_verdicts", {}).items()
        },
    }
