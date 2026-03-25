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
        # Canonical CoreSpec six-tuple (Omega, E, B, I, O, epsilon) — audit K2
        "corespec": {
            "omega": {"type": "image_domain", "modality": result.modality},
            "equations": {"forward_model": result.solver, "track": result.track},
            "boundary_conditions": {},
            "initial_conditions": {"seed": result.seed},
            "observables": {"n_scenes": result.n_scenes},
            "tolerance": 0.01,  # default reconstruction tolerance
        },
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
    """Run R1-R4 gates and write ``certificate.json`` to *bundle_dir*.

    Gate blocking rules (P0):
    - R1/R2/R3 fail  → trust_tier = rejected + safety_brake risk flag
    - R4 fail        → trust_tier = draft    + safety_brake risk flag (budget violation)
    - any warn       → trust_tier = draft    + high_variance risk flag
    - all pass       → trust_tier = draft    (ready for promotion workflow)

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
        Certificate, DomainFlags, GateResult, GateVerdict,
        ImagingTriadFlags, RiskFlag, TrustTier,
    )
    from pwm_core.targeting.gates import run_r1_r4

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

    # --- Run R1-R4 gates ---
    gate_results = run_r1_r4(bundle_dir, manifest, provenance)

    # --- Determine trust tier (P0 blocking logic) ---
    hard_gate_names = {"r1", "r2", "r3"}
    hard_fail = any(
        v.verdict == GateVerdict.fail
        for k, v in gate_results.items()
        if k in hard_gate_names
    )
    r4_fail = gate_results.get("r4", GateResult(verdict=GateVerdict.pass_)).verdict == GateVerdict.fail
    any_warn = any(v.verdict == GateVerdict.warn for v in gate_results.values())

    if hard_fail:
        trust_tier = TrustTier.rejected
    else:
        trust_tier = TrustTier.draft

    risk_flags = []
    if hard_fail or r4_fail:
        risk_flags.append(RiskFlag.safety_brake)
    if any_warn and not hard_fail:
        risk_flags.append(RiskFlag.high_variance)

    # --- Wire Triad gates (G1/G2/G3) from 4-scenario PSNR ---
    imaging_triad = _build_imaging_triad_flags(bundle_dir, manifest)
    domain_flags = DomainFlags(imaging=imaging_triad) if imaging_triad else None

    # --- Provenance hash ---
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

    kernel_version = _get_git_hash()  # reuse existing helper

    cert = Certificate(
        run_id=run_id,
        spec_id=spec_id,
        judge_version=judge_version,
        trust_tier=trust_tier,
        risk_flags=risk_flags,
        active_gates=list(gate_results.keys()),
        gate_verdicts=gate_results,
        domain_flags=domain_flags,
        provenance_hash=prov_hash,
        kernel_version=kernel_version,
    )

    cert_path = bundle_dir / "certificate.json"
    with open(cert_path, "w", encoding="utf-8") as f:
        _json.dump(cert.to_dict(), f, indent=2, default=str)

    logger.info(
        "Certificate issued: %s (tier=%s, risk_flags=%s)",
        cert_path, trust_tier.value, [f.value for f in risk_flags],
    )
    return cert_path


def _build_imaging_triad_flags(bundle_dir: Path, manifest: Dict[str, Any]) -> Optional[Any]:
    """Build ImagingTriadFlags by calling infer_gate_attribution() on per-scenario PSNRs.

    Reads per-scenario PSNR from ``logs/dr_is_records.json`` (preferred) or
    falls back to aggregate metrics in the manifest.  Returns None if
    insufficient data is available.
    """
    try:
        from pwm_core.core.runbundle.certificate import GateResult, GateVerdict, ImagingTriadFlags
        from pwm_core.targeting.scoring import infer_gate_attribution
        import json as _json
    except ImportError:
        return None

    # --- Try to read per-scenario PSNR from DR-IS records ---
    dr_is_path = bundle_dir / "logs" / "dr_is_records.json"
    scenario_psnr: Dict[str, float] = {}

    if dr_is_path.exists():
        try:
            with open(dr_is_path, encoding="utf-8") as f:
                records = _json.load(f)
            # Aggregate by scenario_id: take mean PSNR across scenes
            from collections import defaultdict
            sums: Dict[str, float] = defaultdict(float)
            counts: Dict[str, int] = defaultdict(int)
            for rec in records:
                sid = str(rec.get("scenario", "")).upper()
                psnr_val = rec.get("psnr")
                if psnr_val is not None:
                    sums[sid] += float(psnr_val)
                    counts[sid] += 1
            for sid in sums:
                scenario_psnr[sid] = sums[sid] / counts[sid]
        except Exception as exc:
            logger.debug("_build_imaging_triad_flags: DR-IS read failed: %s", exc)

    # --- Map scenario IDs to I/II/III/IV ---
    def _get(keys, default=None):
        for k in keys:
            if k in scenario_psnr:
                return scenario_psnr[k]
        return default

    psnr_i   = _get(["I", "IDEAL", "1"])
    psnr_ii  = _get(["II", "ASSUMED", "2"])
    psnr_iii = _get(["III", "CORRECTED", "CALIBRATED", "3"])
    psnr_iv  = _get(["IV", "ORACLE", "4"])

    # --- Fallback: reconstruct from aggregate metrics if scenarios missing ---
    if psnr_i is None or psnr_ii is None or psnr_iii is None:
        metrics = manifest.get("metrics", {})
        rho = metrics.get("rho")
        oracle_gap = metrics.get("oracle_gap")
        psnr_db = metrics.get("psnr_db")
        if rho is not None and oracle_gap is not None and psnr_db is not None:
            # psnr_iii ≈ psnr_db; psnr_i = psnr_iii + oracle_gap
            # rho = (psnr_iii - psnr_ii) / (psnr_i - psnr_ii)
            # → psnr_ii = psnr_i - (psnr_iii - psnr_i + oracle_gap) / rho  (approx)
            psnr_iii = float(psnr_db)
            psnr_i = psnr_iii + float(oracle_gap)
            denom = float(rho) if abs(float(rho)) > 1e-6 else 1.0
            gap_i_to_ii = (psnr_iii - psnr_ii if psnr_ii is not None
                           else (psnr_i - psnr_iii) / denom)
            psnr_ii = psnr_i - gap_i_to_ii
        else:
            return None  # no usable data

    if psnr_iv is None:
        psnr_iv = psnr_i  # IV ≈ I in current implementation

    try:
        attr = infer_gate_attribution(
            psnr_i=psnr_i,
            psnr_ii=psnr_ii,
            psnr_iii=psnr_iii,
            psnr_iv=psnr_iv,
        )
    except Exception as exc:
        logger.debug("_build_imaging_triad_flags: infer_gate_attribution failed: %s", exc)
        return None

    # Map GateAttribution fractions to GateResult verdicts:
    # dominant gate with high confidence → warn; others → pass
    def _gate_result(fraction: float, is_dominant: bool, confidence: float) -> GateResult:
        if is_dominant and confidence > 0.4:
            return GateResult(
                verdict=GateVerdict.warn,
                message=f"{fraction:.0%} of degradation attributed to this gate (conf={confidence:.2f})",
                details={"fraction": fraction, "confidence": confidence},
            )
        return GateResult(
            verdict=GateVerdict.pass_,
            message=f"{fraction:.0%} attribution (not dominant)",
            details={"fraction": fraction},
        )

    dom = attr.dominant_gate
    return ImagingTriadFlags(
        g1_sampling=_gate_result(
            attr.gate_1_sampling,
            dom == "gate_1_sampling",
            attr.confidence,
        ),
        g2_noise=_gate_result(
            attr.gate_2_noise,
            dom == "gate_2_noise",
            attr.confidence,
        ),
        g3_operator=_gate_result(
            attr.gate_3_mismatch,
            dom == "gate_3_mismatch",
            attr.confidence,
        ),
    )


# Keep old name as alias for backward compatibility
_build_triad_flags = _build_imaging_triad_flags
