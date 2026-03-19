"""
Runs Router — submit, list, and inspect PWM pipeline runs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from pwm_platform.auth.dependencies import get_current_user
from pwm_platform.db.database import async_session_factory, get_db
from pwm_platform.db.models import Run, TriadReport, User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/runs", tags=["Runs"])


# ── Schemas ──────────────────────────────────────────────────────────────


class RunCreateRequest(BaseModel):
    modality: str
    task_kind: str = "simulate_recon_analyze"
    compute_mode: str = "auto"
    input_mode: str = "spec"
    experiment_spec: dict = {}
    dataset_id: Optional[str] = None
    prompt: Optional[str] = None


class RunListResponse(BaseModel):
    runs: list[dict]
    total: int


# ── Mock execution (demo) ───────────────────────────────────────────────

# Realistic metrics per modality — all 64 modalities
# Keys: psnr (dB range), ssim range, sam (spectral angle, optional), solver, iters range
_MODALITY_METRICS = {
    # ── Compressive ───────────────────────────────────────────────────────
    "cassi":             {"psnr": (28, 36), "ssim": (0.88, 0.96), "sam": (0.03, 0.08), "solver": "mst",             "iters": (80, 200)},
    "cacti":             {"psnr": (26, 34), "ssim": (0.86, 0.94), "sam": None,          "solver": "gap_tv",          "iters": (80, 200)},
    "spc":               {"psnr": (22, 32), "ssim": (0.80, 0.92), "sam": None,          "solver": "pnp_fista",       "iters": (100, 250)},
    "matrix":            {"psnr": (24, 34), "ssim": (0.82, 0.93), "sam": None,          "solver": "fista_l2",        "iters": (80, 200)},
    # ── Medical ───────────────────────────────────────────────────────────
    "ct":                {"psnr": (32, 42), "ssim": (0.92, 0.99), "sam": None,          "solver": "fbp",             "iters": (1, 1)},
    "cbct":              {"psnr": (30, 40), "ssim": (0.90, 0.97), "sam": None,          "solver": "fdk",             "iters": (1, 1)},
    "mri":               {"psnr": (30, 40), "ssim": (0.90, 0.97), "sam": None,          "solver": "sense",           "iters": (50, 150)},
    "fmri":              {"psnr": (28, 36), "ssim": (0.86, 0.94), "sam": None,          "solver": "sense",           "iters": (30, 100)},
    "diffusion_mri":     {"psnr": (26, 35), "ssim": (0.84, 0.93), "sam": None,          "solver": "weighted_least_squares", "iters": (20, 60)},
    "mrs":               {"psnr": (20, 30), "ssim": (0.75, 0.88), "sam": None,          "solver": "lcmodel",         "iters": (50, 150)},
    "pet":               {"psnr": (24, 33), "ssim": (0.82, 0.92), "sam": None,          "solver": "mlem",            "iters": (30, 80)},
    "spect":             {"psnr": (22, 31), "ssim": (0.78, 0.90), "sam": None,          "solver": "mlem",            "iters": (30, 80)},
    "ultrasound":        {"psnr": (26, 35), "ssim": (0.84, 0.93), "sam": None,          "solver": "tv_fista",        "iters": (30, 100)},
    "doppler_ultrasound":{"psnr": (24, 33), "ssim": (0.80, 0.91), "sam": None,          "solver": "autocorrelation_estimator", "iters": (1, 1)},
    "elastography":      {"psnr": (22, 31), "ssim": (0.78, 0.89), "sam": None,          "solver": "time_of_flight_inversion", "iters": (20, 60)},
    "fluoroscopy":       {"psnr": (28, 37), "ssim": (0.86, 0.95), "sam": None,          "solver": "tv_fista",        "iters": (20, 60)},
    "angiography":       {"psnr": (30, 39), "ssim": (0.88, 0.96), "sam": None,          "solver": "dsa_subtraction", "iters": (1, 1)},
    "xray_radiography":  {"psnr": (30, 40), "ssim": (0.90, 0.97), "sam": None,          "solver": "tv_fista",        "iters": (10, 40)},
    "mammography":       {"psnr": (29, 38), "ssim": (0.88, 0.96), "sam": None,          "solver": "tv_fista",        "iters": (15, 50)},
    "dexa":              {"psnr": (28, 37), "ssim": (0.86, 0.95), "sam": None,          "solver": "dual_energy_decomposition", "iters": (1, 1)},
    "dot":               {"psnr": (18, 27), "ssim": (0.68, 0.82), "sam": None,          "solver": "born_approx",     "iters": (50, 150)},
    "photoacoustic":     {"psnr": (24, 33), "ssim": (0.80, 0.91), "sam": None,          "solver": "back_projection", "iters": (1, 1)},
    # ── Coherent ──────────────────────────────────────────────────────────
    "ptychography":      {"psnr": (25, 35), "ssim": (0.85, 0.95), "sam": None,          "solver": "epie",            "iters": (100, 300)},
    "holography":        {"psnr": (27, 37), "ssim": (0.87, 0.96), "sam": None,          "solver": "angular_spectrum", "iters": (1, 1)},
    "phase_retrieval":   {"psnr": (23, 33), "ssim": (0.82, 0.93), "sam": None,          "solver": "hio",             "iters": (100, 300)},
    # ── Microscopy ────────────────────────────────────────────────────────
    "widefield":         {"psnr": (28, 37), "ssim": (0.86, 0.95), "sam": None,          "solver": "richardson_lucy",  "iters": (20, 60)},
    "widefield_lowdose": {"psnr": (24, 33), "ssim": (0.80, 0.91), "sam": None,          "solver": "pnp_hqs",         "iters": (30, 80)},
    "confocal_livecell": {"psnr": (29, 38), "ssim": (0.88, 0.96), "sam": None,          "solver": "richardson_lucy", "iters": (20, 50)},
    "confocal_3d":       {"psnr": (28, 37), "ssim": (0.86, 0.95), "sam": None,          "solver": "richardson_lucy_3d", "iters": (20, 60)},
    "sim":               {"psnr": (30, 39), "ssim": (0.90, 0.97), "sam": None,          "solver": "wiener_sim",      "iters": (1, 1)},
    "lightsheet":        {"psnr": (29, 38), "ssim": (0.88, 0.96), "sam": None,          "solver": "fourier_notch_destripe", "iters": (1, 1)},
    "two_photon":        {"psnr": (27, 36), "ssim": (0.85, 0.94), "sam": None,          "solver": "richardson_lucy", "iters": (20, 60)},
    "sted":              {"psnr": (26, 35), "ssim": (0.84, 0.93), "sam": None,          "solver": "richardson_lucy", "iters": (20, 60)},
    "tirf":              {"psnr": (28, 37), "ssim": (0.86, 0.95), "sam": None,          "solver": "richardson_lucy", "iters": (20, 50)},
    "flim":              {"psnr": (22, 31), "ssim": (0.78, 0.89), "sam": None,          "solver": "phasor",          "iters": (1, 1)},
    "fpm":               {"psnr": (26, 35), "ssim": (0.84, 0.93), "sam": None,          "solver": "sequential_phase_retrieval", "iters": (50, 150)},
    "palm_storm":        {"psnr": (20, 30), "ssim": (0.75, 0.88), "sam": None,          "solver": "thunderstorm",    "iters": (1, 1)},
    "polarization":      {"psnr": (27, 36), "ssim": (0.85, 0.94), "sam": None,          "solver": "pnp_hqs",         "iters": (30, 80)},
    # ── Electron Microscopy ───────────────────────────────────────────────
    "sem":               {"psnr": (30, 40), "ssim": (0.90, 0.97), "sam": None,          "solver": "direct_imaging",  "iters": (1, 1)},
    "tem":               {"psnr": (28, 38), "ssim": (0.88, 0.96), "sam": None,          "solver": "ctf_correction",  "iters": (1, 1)},
    "stem":              {"psnr": (29, 39), "ssim": (0.89, 0.97), "sam": None,          "solver": "direct_imaging",  "iters": (1, 1)},
    "electron_tomography": {"psnr": (26, 35), "ssim": (0.84, 0.93), "sam": None,        "solver": "sirt",            "iters": (30, 80)},
    "electron_diffraction": {"psnr": (24, 33), "ssim": (0.82, 0.92), "sam": None,       "solver": "ptychography_epie", "iters": (80, 200)},
    "electron_holography": {"psnr": (26, 36), "ssim": (0.85, 0.94), "sam": None,        "solver": "fourier_sideband", "iters": (1, 1)},
    "ebsd":              {"psnr": (28, 37), "ssim": (0.86, 0.95), "sam": None,          "solver": "hough_indexing",  "iters": (1, 1)},
    "eels":              {"psnr": (22, 31), "ssim": (0.78, 0.89), "sam": None,          "solver": "fourier_ratio",   "iters": (1, 1)},
    # ── Clinical Optics ───────────────────────────────────────────────────
    "oct":               {"psnr": (28, 38), "ssim": (0.87, 0.96), "sam": None,          "solver": "fft_recon",       "iters": (1, 1)},
    "octa":              {"psnr": (26, 35), "ssim": (0.84, 0.93), "sam": None,          "solver": "tv_fista",        "iters": (20, 60)},
    "fundus":            {"psnr": (30, 39), "ssim": (0.90, 0.97), "sam": None,          "solver": "richardson_lucy", "iters": (10, 30)},
    "endoscopy":         {"psnr": (26, 35), "ssim": (0.84, 0.93), "sam": None,          "solver": "tv_fista",        "iters": (20, 60)},
    # ── Computational / Computational Photography ─────────────────────────
    "light_field":       {"psnr": (28, 37), "ssim": (0.86, 0.95), "sam": None,          "solver": "shift_and_sum",   "iters": (1, 1)},
    "integral":          {"psnr": (27, 36), "ssim": (0.85, 0.94), "sam": None,          "solver": "depth_estimation", "iters": (1, 1)},
    "lensless":          {"psnr": (22, 31), "ssim": (0.78, 0.89), "sam": None,          "solver": "admm_tv",         "iters": (50, 150)},
    "panorama":          {"psnr": (30, 39), "ssim": (0.90, 0.97), "sam": None,          "solver": "laplacian_pyramid_fusion", "iters": (1, 1)},
    # ── Neural Rendering ──────────────────────────────────────────────────
    "nerf":              {"psnr": (26, 34), "ssim": (0.84, 0.93), "sam": None,          "solver": "nerf_mlp",        "iters": (5000, 30000)},
    "gaussian_splatting": {"psnr": (28, 36), "ssim": (0.88, 0.96), "sam": None,         "solver": "gaussian_splatting_3dgs", "iters": (3000, 15000)},
    # ── Depth Imaging ─────────────────────────────────────────────────────
    "tof_camera":        {"psnr": (28, 37), "ssim": (0.86, 0.95), "sam": None,          "solver": "tv_fista",        "iters": (10, 40)},
    "structured_light":  {"psnr": (30, 39), "ssim": (0.90, 0.97), "sam": None,          "solver": "phase_unwrap",    "iters": (1, 1)},
    "lidar":             {"psnr": (26, 35), "ssim": (0.84, 0.93), "sam": None,          "solver": "tv_fista",        "iters": (10, 40)},
    # ── Remote Sensing ────────────────────────────────────────────────────
    "sar":               {"psnr": (24, 33), "ssim": (0.80, 0.91), "sam": None,          "solver": "backprojection",  "iters": (1, 1)},
    "sonar":             {"psnr": (22, 31), "ssim": (0.76, 0.88), "sam": None,          "solver": "beamform_das",    "iters": (1, 1)},
    # ── Particle Imaging ──────────────────────────────────────────────────
    "neutron_tomo":      {"psnr": (26, 35), "ssim": (0.84, 0.93), "sam": None,          "solver": "filtered_back_projection", "iters": (1, 1)},
    "proton_radiography": {"psnr": (24, 33), "ssim": (0.80, 0.91), "sam": None,         "solver": "filtered_back_projection", "iters": (1, 1)},
    "muon_tomo":         {"psnr": (20, 29), "ssim": (0.72, 0.85), "sam": None,          "solver": "poca_reconstruction", "iters": (30, 80)},
}


async def _mock_execute_run(run_id: str, modality: str, compute_mode: str = "auto"):
    """Simulate a run execution in the background (demo mode)."""
    from pwm_platform.services.demo_images import generate_demo_images

    # GPU mode runs faster
    is_gpu = compute_mode == "gpu" or (compute_mode == "auto" and modality in ("cassi", "mri"))
    speed = 0.5 if is_gpu else 1.0

    await asyncio.sleep(1.0 * speed)  # simulate queue delay

    async with async_session_factory() as db:
        result = await db.execute(select(Run).where(Run.run_id == run_id))
        run = result.scalar_one_or_none()
        if run is None:
            return

        worker = "gpu-pytorch" if is_gpu else "cpu-local"

        # Step 1: Initializing
        run.status = "initializing"
        run.started_at = datetime.now(timezone.utc)
        run.worker_type = worker
        await db.commit()
        await asyncio.sleep(random.uniform(0.8, 1.5) * speed)

        # Step 2: Simulating forward model
        run.status = "simulating"
        await db.commit()
        await asyncio.sleep(random.uniform(1.5, 3.0) * speed)

        # Step 3: Reconstructing (GPU-accelerated solvers run faster)
        run.status = "reconstructing"
        await db.commit()
        await asyncio.sleep(random.uniform(2.0, 4.0) * speed)

        # Step 4: Analyzing + generating output images
        run.status = "analyzing"
        await db.commit()

        # Generate realistic metrics
        m = _MODALITY_METRICS.get(modality, _MODALITY_METRICS["cassi"])
        # GPU mode gets slightly better metrics (more iterations converge better)
        psnr_boost = 1.5 if is_gpu else 0
        psnr = round(random.uniform(*m["psnr"]) + psnr_boost, 2)
        ssim = round(min(random.uniform(*m["ssim"]) + (0.01 if is_gpu else 0), 0.9999), 4)
        sam = round(random.uniform(*m["sam"]), 4) if m["sam"] else None
        lpips = round(random.uniform(0.01, 0.15), 4)
        iters = random.randint(*m["iters"])
        if is_gpu:
            iters = int(iters * 1.5)  # GPU can afford more iterations

        # Generate demo images (run in thread to avoid blocking event loop)
        loop = asyncio.get_event_loop()
        image_urls = await loop.run_in_executor(
            None, generate_demo_images, run_id, modality, psnr
        )

        await asyncio.sleep(random.uniform(0.5, 1.5) * speed)

        # Calculate total duration
        duration = (datetime.now(timezone.utc) - run.started_at).total_seconds()

        # Solver name — GPU uses PyTorch-accelerated variant
        solver = m["solver"]
        if is_gpu:
            solver = f"{solver}_torch"

        # Step 5: Completed
        run.status = "completed"
        run.completed_at = datetime.now(timezone.utc)
        run.duration_seconds = round(duration, 1)
        await db.commit()

        # Create TriadReport with image URLs in custom_metrics
        report = TriadReport(
            run_id=run_id,
            modality=modality,
            psnr=psnr,
            ssim=ssim,
            lpips=lpips,
            sam=sam,
            custom_metrics={"images": image_urls} if image_urls else {},
            diagnosis_severity="info" if psnr > 30 else "warning",
            diagnosis_codes=["quality_good"] if psnr > 30 else ["low_psnr", "check_calibration"],
            recommended_actions=[] if psnr > 30 else ["increase_iterations", "verify_forward_model"],
            uncertainty_mean=round(random.uniform(0.001, 0.01), 4),
            uncertainty_max=round(random.uniform(0.01, 0.05), 4),
            confidence_interval={"lower": round(psnr - 0.5, 2), "upper": round(psnr + 0.5, 2)},
            operator_mismatch_detected=random.random() < 0.15,
            mismatch_type="calibration_drift" if random.random() < 0.15 else None,
            reconstruction_method=solver,
            solver_iterations=iters,
            convergence_residual=round(random.uniform(1e-6, 1e-3), 8),
        )
        db.add(report)
        await db.commit()

        logger.info("Run %s completed: PSNR=%.2f SSIM=%.4f worker=%s", run_id, psnr, ssim, worker)


# ── Endpoints ────────────────────────────────────────────────────────────


@router.post("", response_model=dict)
async def create_run(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Submit a new pipeline run. Accepts both JSON and form data."""
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        data = await request.json()
    else:
        # Form data from HTMX
        form = await request.form()
        spec_text = form.get("spec_json", "")
        try:
            experiment_spec = json.loads(spec_text) if spec_text.strip().startswith("{") else {}
        except (json.JSONDecodeError, ValueError):
            experiment_spec = {}

        data = {
            "modality": form.get("modality", "cassi"),
            "task_kind": form.get("task_kind", "simulate_recon_analyze"),
            "compute_mode": form.get("compute_mode", "auto"),
            "input_mode": form.get("input_mode", "spec"),
            "experiment_spec": experiment_spec,
            "prompt": spec_text if not spec_text.strip().startswith("{") else None,
            "is_public": form.get("is_public", "true"),
        }

    # ── Auto-detect modality from prompt when set to "auto" ─────────────
    prompt_text = data.get("prompt") or ""
    if not prompt_text:
        # Also check spec_json (used by both form and JSON submissions)
        spec_json_text = data.get("spec_json", "")
        if isinstance(spec_json_text, str) and spec_json_text.strip() and not spec_json_text.strip().startswith("{"):
            prompt_text = spec_json_text.strip()
    if not prompt_text:
        # Also check experiment_spec
        spec = data.get("experiment_spec", {})
        if isinstance(spec, dict):
            prompt_text = spec.get("prompt", "")

    modality = data.get("modality", "cassi")
    if modality == "auto" and prompt_text:
        from pwm_platform.services.modality_router import detect_modality
        detected, confidence = detect_modality(prompt_text)
        modality = detected
        logger.info(
            "Auto-detected modality: %s (confidence=%.2f) from prompt: %s",
            detected, confidence, prompt_text[:100],
        )
    elif modality == "auto":
        modality = "cassi"  # fallback when no prompt given

    # Auto-populate default experimental setup for the detected modality
    from pwm_platform.services.experiment_defaults import get_default_experiment_spec
    default_spec = get_default_experiment_spec(modality)

    # Merge: default spec is the base, user-provided spec overrides
    experiment_spec = data.get("experiment_spec", {})
    if not isinstance(experiment_spec, dict):
        experiment_spec = {}
    merged_spec = {**default_spec, **experiment_spec}
    if prompt_text:
        merged_spec["prompt"] = prompt_text
    experiment_spec = merged_spec

    run_id = f"run-{uuid.uuid4().hex[:12]}"

    is_public_raw = data.get("is_public", "true")
    is_public = str(is_public_raw).lower() not in ("false", "0", "no", "private")

    run = Run(
        run_id=run_id,
        user_id=user.id,
        modality=modality,
        task_kind=data.get("task_kind", "simulate_recon_analyze"),
        status="pending",
        compute_mode=data.get("compute_mode", "auto"),
        input_mode=data.get("input_mode", "prompt"),
        experiment_spec=experiment_spec,
        dataset_id=data.get("dataset_id"),
        is_public=is_public,
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    logger.info("Created run %s for user %s (%s / %s)", run_id, user.id, run.modality, run.task_kind)

    # Start mock execution in background
    asyncio.create_task(_mock_execute_run(run_id, run.modality, run.compute_mode))

    # If HTML request (HTMX form), redirect to run status page
    accept = request.headers.get("accept", "")
    if "text/html" in accept or "application/x-www-form-urlencoded" in content_type:
        return RedirectResponse(f"/runs/{run_id}", status_code=303)

    return {
        "success": True,
        "run_id": run_id,
        "status": "pending",
        "message": "Run submitted",
    }


@router.get("", response_model=RunListResponse)
async def list_runs(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = 50,
    offset: int = 0,
    visibility: Optional[str] = None,
):
    """List runs for the current user. Optional visibility filter: public/private."""
    stmt = (
        select(Run)
        .where(Run.user_id == user.id)
        .order_by(Run.submitted_at.desc())
        .limit(limit)
        .offset(offset)
    )
    if visibility == "public":
        stmt = stmt.where(Run.is_public == True)  # noqa: E712
    elif visibility == "private":
        stmt = stmt.where(Run.is_public == False)  # noqa: E712

    result = await db.execute(stmt)
    runs = result.scalars().all()

    return {
        "runs": [
            {
                "run_id": r.run_id,
                "modality": r.modality,
                "task_kind": r.task_kind,
                "status": r.status,
                "compute_mode": r.compute_mode,
                "is_public": r.is_public,
                "submitted_at": r.submitted_at.isoformat() if r.submitted_at else None,
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                "duration_seconds": r.duration_seconds,
            }
            for r in runs
        ],
        "total": len(runs),
    }


@router.get("/{run_id}")
async def get_run(
    run_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get run details. Public runs visible to all; private runs only to owner/admin."""
    result = await db.execute(select(Run).where(Run.run_id == run_id))
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    # Private runs: only owner or admin
    if not run.is_public and run.user_id != user.id and user.role != "admin":
        raise HTTPException(status_code=404, detail="Run not found")

    return {
        "run_id": run.run_id,
        "modality": run.modality,
        "task_kind": run.task_kind,
        "status": run.status,
        "compute_mode": run.compute_mode,
        "is_public": run.is_public,
        "input_mode": run.input_mode,
        "experiment_spec": run.experiment_spec,
        "submitted_at": run.submitted_at.isoformat() if run.submitted_at else None,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "duration_seconds": run.duration_seconds,
        "worker_type": run.worker_type,
        "error_message": run.error_message,
        "gcs_bundle_path": run.gcs_bundle_path,
    }


@router.get("/{run_id}/results")
async def get_run_results(
    run_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get TriadReport results for a run. Public runs visible to all; private runs only to owner/admin."""
    run_result = await db.execute(select(Run).where(Run.run_id == run_id))
    run = run_result.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    # Private runs: only owner or admin
    if not run.is_public and run.user_id != user.id and user.role != "admin":
        raise HTTPException(status_code=404, detail="Run not found")

    report_result = await db.execute(
        select(TriadReport).where(TriadReport.run_id == run_id)
    )
    report = report_result.scalar_one_or_none()
    if report is None:
        raise HTTPException(status_code=404, detail="Results not available yet")

    return {
        "run_id": run_id,
        "modality": report.modality,
        "psnr": report.psnr,
        "ssim": report.ssim,
        "lpips": report.lpips,
        "sam": report.sam,
        "custom_metrics": report.custom_metrics,
        "diagnosis_severity": report.diagnosis_severity,
        "diagnosis_codes": report.diagnosis_codes,
        "recommended_actions": report.recommended_actions,
        "uncertainty_mean": report.uncertainty_mean,
        "uncertainty_max": report.uncertainty_max,
        "confidence_interval": report.confidence_interval,
        "operator_mismatch_detected": report.operator_mismatch_detected,
        "mismatch_type": report.mismatch_type,
        "reconstruction_method": report.reconstruction_method,
        "solver_iterations": report.solver_iterations,
    }


@router.patch("/{run_id}/visibility")
async def update_run_visibility(
    run_id: str,
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Toggle run visibility (public/private). Owner only."""
    result = await db.execute(select(Run).where(Run.run_id == run_id))
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.user_id != user.id and user.role != "admin":
        raise HTTPException(status_code=403, detail="Only the owner can change visibility")

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        data = await request.json()
        is_public_raw = data.get("is_public", not run.is_public)
    else:
        form = await request.form()
        is_public_raw = form.get("is_public", str(not run.is_public))

    run.is_public = str(is_public_raw).lower() not in ("false", "0", "no", "private")
    await db.commit()

    # For HTMX requests, return an HTML snippet with the updated badge + toggle
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        badge_color = "green" if run.is_public else "gray"
        badge_text = "Public" if run.is_public else "Private"
        next_action = "false" if run.is_public else "true"
        next_label = "Make Private" if run.is_public else "Make Public"
        html = (
            f'<span id="visibility-controls" class="flex items-center space-x-2">'
            f'<span class="px-2 py-0.5 text-xs font-semibold rounded-full bg-{badge_color}-100 text-{badge_color}-800">{badge_text}</span>'
            f'<button hx-patch="/api/v1/runs/{run.run_id}/visibility" '
            f'hx-vals=\'{{"is_public": "{next_action}"}}\' '
            f'hx-headers=\'{{"Content-Type": "application/json"}}\' '
            f'hx-target="#visibility-controls" hx-swap="outerHTML" '
            f'class="text-xs text-indigo-600 hover:text-indigo-800 underline">{next_label}</button>'
            f'</span>'
        )
        from fastapi.responses import HTMLResponse
        return HTMLResponse(html)

    return {"run_id": run.run_id, "is_public": run.is_public}
