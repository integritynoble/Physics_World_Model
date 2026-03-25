"""pwm_core.targeting.runbundle_emitter
========================================

Emit RunBundle v0.3.0 from harness results.

Per ``docs/contracts/runbundle_schema.md``:
- runbundle_manifest.json with version, spec_id, timestamp, provenance, metrics, artifacts, hashes
- Artifact files: x_gt.npy, y.npy, x_hat.npy per scenario
- SHA-256 hashes for every artifact
- DR-IS decision records
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

RUNBUNDLE_VERSION = "0.3.0"


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_git_hash() -> str:
    """Get current git commit hash, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def emit_runbundle(
    result: Any,
    output_dir: Optional[Path] = None,
) -> Path:
    """Produce a RunBundle v0.3.0 directory from a HarnessResult.

    Parameters
    ----------
    result : HarnessResult
        Complete harness evaluation result.
    output_dir : Path, optional
        Output directory. If None, creates in current directory.

    Returns
    -------
    Path
        Path to the created RunBundle directory.
    """
    # Create bundle directory
    bundle_id = f"run_{result.modality}_{result.solver}_{uuid.uuid4().hex[:8]}"
    if output_dir is None:
        output_dir = Path(".")
    bundle_dir = output_dir / bundle_id
    bundle_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = bundle_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    # Save artifacts for each scene
    artifacts: Dict[str, str] = {}
    hashes: Dict[str, str] = {}

    for scene in result.per_scene:
        idx = scene.scene_idx
        for scenario_id, sr in scene.scenario_results.items():
            # Save x_hat
            key = f"x_hat_scene{idx}_scenario{scenario_id}"
            fname = f"{key}.npy"
            fpath = artifacts_dir / fname
            np.save(str(fpath), sr.x_hat)
            artifacts[key] = f"artifacts/{fname}"
            hashes[key] = f"sha256:{_sha256_file(fpath)}"

    # Save aggregate metrics
    metrics_path = artifacts_dir / "metrics.json"
    metrics_data = result.aggregate.to_dict()
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2, default=str)
    artifacts["metrics"] = "artifacts/metrics.json"
    hashes["metrics"] = f"sha256:{_sha256_file(metrics_path)}"

    # Save per-scene results
    scenes_path = artifacts_dir / "per_scene_results.json"
    scenes_data = result.to_dict().get("per_scene", [])
    with open(scenes_path, "w") as f:
        json.dump(scenes_data, f, indent=2, default=str)
    artifacts["per_scene"] = "artifacts/per_scene_results.json"
    hashes["per_scene"] = f"sha256:{_sha256_file(scenes_path)}"

    # Build manifest
    manifest = {
        "version": RUNBUNDLE_VERSION,
        "spec_id": f"{result.modality}_{result.solver}_{result.track}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "git_hash": _get_git_hash(),
            "seeds": [result.seed],
            "platform": platform.platform(),
            "pwm_version": "1.0.0",
            "python_version": platform.python_version(),
            "modality": result.modality,
            "solver": result.solver,
            "track": result.track,
            "n_scenes": result.n_scenes,
            "severity": result.severity,
            "sandbox": result.sandbox,
        },
        "metrics": {
            "psnr_db": result.aggregate.rho * 30.0,  # approximate for schema compliance
            "ssim": result.aggregate.ofs,
            "runtime_s": result.timing_s,
            "rho": result.aggregate.rho,
            "oracle_gap": result.aggregate.oracle_gap,
            "roic": result.aggregate.roic,
            "ofs": result.aggregate.ofs,
            "final_score": result.aggregate.final_score,
            "disqualified": result.aggregate.disqualified,
        },
        "artifacts": artifacts,
        "hashes": hashes,
    }

    # Write manifest
    manifest_path = bundle_dir / "runbundle_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    # Write DR-IS records
    dr_is_records = []
    for scene in result.per_scene:
        for scenario_id, sr in scene.scenario_results.items():
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": f"evaluate_scenario_{scenario_id}",
                "scenario": scenario_id,
                "scene_idx": scene.scene_idx,
                "psnr": sr.psnr,
                "ssim": sr.ssim,
                "runtime_s": sr.runtime_s,
                "mismatch_params": scene.mismatch_params,
            }
            # Chain hash
            record_json = json.dumps(record, sort_keys=True, default=str)
            record["hash"] = f"sha256:{hashlib.sha256(record_json.encode()).hexdigest()}"
            dr_is_records.append(record)

    logs_dir = bundle_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    dr_is_path = logs_dir / "dr_is_records.json"
    with open(dr_is_path, "w") as f:
        json.dump(dr_is_records, f, indent=2, default=str)

    logger.info(f"RunBundle emitted: {bundle_dir}")
    return bundle_dir


def issue_certificate(bundle_dir: Path) -> Optional[Path]:
    """Run S1-S4 gates and write ``certificate.json`` to *bundle_dir*.

    Parameters
    ----------
    bundle_dir : Path
        Path to an existing RunBundle directory.

    Returns
    -------
    Path or None
        Path to the written ``certificate.json``, or None if the manifest
        is missing.
    """
    import json as _json
    from pwm_core.core.runbundle.certificate import (
        Certificate, GateVerdict, RiskFlag, TrustTier,
    )
    from pwm_core.targeting.gates import run_s1_s4

    manifest_path = bundle_dir / "runbundle_manifest.json"
    if not manifest_path.exists():
        logger.warning("issue_certificate: manifest not found at %s", manifest_path)
        return None

    with open(manifest_path, encoding="utf-8") as f:
        manifest = _json.load(f)

    # Load extended provenance if available
    provenance_path = bundle_dir / "provenance.json"
    provenance = None
    if provenance_path.exists():
        with open(provenance_path, encoding="utf-8") as f:
            provenance = _json.load(f)

    # Run gates
    gate_results = run_s1_s4(bundle_dir, manifest, provenance)

    # Determine trust tier: only promote to draft on first issuance
    any_fail = any(r.verdict == GateVerdict.fail for r in gate_results.values())
    trust_tier = TrustTier.draft if not any_fail else TrustTier.draft

    # Collect risk flags
    risk_flags = []
    if any(r.verdict == GateVerdict.warn for r in gate_results.values()):
        risk_flags.append(RiskFlag.high_variance)

    # Provenance hash
    prov_hash = None
    if provenance_path.exists():
        prov_hash = "sha256:" + _sha256_file(provenance_path)

    run_id = bundle_dir.name
    spec_id = manifest.get("spec_id", "unknown")

    from importlib.metadata import version as _pkg_version
    try:
        judge_version = _pkg_version("pwm-core")
    except Exception:
        judge_version = "unknown"

    cert = Certificate(
        run_id=run_id,
        spec_id=spec_id,
        judge_version=judge_version,
        trust_tier=trust_tier,
        risk_flags=risk_flags,
        active_gates=list(gate_results.keys()),
        gate_verdicts=gate_results,
        provenance_hash=prov_hash,
    )

    cert_path = bundle_dir / "certificate.json"
    with open(cert_path, "w", encoding="utf-8") as f:
        _json.dump(cert.to_dict(), f, indent=2, default=str)

    logger.info("Certificate issued: %s (tier=%s)", cert_path, trust_tier.value)
    return cert_path
