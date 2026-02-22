"""
Pages Router — server-rendered HTML pages (Jinja2 + HTMX).

Public pages: all viewing pages (dashboard, datasets, modalities, run status).
Auth-required: actions that run PWM reconstruction (new run, bootstrap, review).
Login: CompareGPT SSO redirect flow.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Optional

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from pwm_platform.auth.dependencies import get_current_user, get_optional_user
from pwm_platform.auth.service import auth_service
from pwm_platform.config import settings
from pwm_platform.db.database import get_db
from pwm_platform.db.models import (
    BootstrapProposal,
    Dataset,
    ModalityBasics,
    Run,
    TriadReport,
    User,
)


# ── Fallback modality list (used when DB is empty / not yet seeded) ───────
# Format: (modality_key, display_name, category)
_FALLBACK_MODALITIES_RAW = [
    # Compressive
    ("cassi", "CASSI (Coded Aperture Snapshot Spectral Imaging)", "compressive"),
    ("cacti", "CACTI (Coded Aperture Compressive Temporal Imaging)", "compressive"),
    ("spc", "Single-Pixel Camera", "compressive"),
    ("matrix", "Generic Matrix Sensing", "compressive"),
    # Medical
    ("ct", "CT (X-ray Computed Tomography)", "medical"),
    ("cbct", "Cone-Beam CT (CBCT)", "medical"),
    ("mri", "MRI (Magnetic Resonance Imaging)", "medical"),
    ("fmri", "Functional MRI (BOLD)", "medical"),
    ("diffusion_mri", "Diffusion MRI (DTI)", "medical"),
    ("mrs", "MR Spectroscopy", "medical"),
    ("pet", "Positron Emission Tomography (PET)", "medical"),
    ("spect", "Single Photon Emission CT (SPECT)", "medical"),
    ("ultrasound", "Ultrasound Imaging", "medical"),
    ("doppler_ultrasound", "Doppler Ultrasound", "medical_ultrasound"),
    ("elastography", "Shear-Wave Elastography", "medical_ultrasound"),
    ("fluoroscopy", "Fluoroscopy", "medical"),
    ("angiography", "X-ray Angiography", "medical"),
    ("xray_radiography", "X-ray Radiography", "medical"),
    ("mammography", "Mammography", "medical"),
    ("dexa", "Dual-Energy X-ray Absorptiometry (DEXA)", "medical"),
    ("dot", "Diffuse Optical Tomography", "medical"),
    ("photoacoustic", "Photoacoustic Imaging", "medical"),
    # Coherent
    ("holography", "Digital Holographic Microscopy", "coherent"),
    ("ptychography", "Ptychography", "coherent"),
    ("phase_retrieval", "Coherent Diffractive Imaging (CDI)", "coherent"),
    # Microscopy
    ("widefield", "Widefield Fluorescence Microscopy", "microscopy"),
    ("widefield_lowdose", "Low-Dose Widefield Microscopy", "microscopy"),
    ("confocal_livecell", "Confocal Live-Cell Microscopy", "microscopy"),
    ("confocal_3d", "Confocal 3D Z-Stack", "microscopy"),
    ("sim", "Structured Illumination Microscopy (SIM)", "microscopy"),
    ("lightsheet", "Light-Sheet Fluorescence Microscopy", "microscopy"),
    ("two_photon", "Two-Photon / Multiphoton Microscopy", "microscopy"),
    ("sted", "STED Microscopy", "microscopy"),
    ("tirf", "TIRF Microscopy", "microscopy"),
    ("flim", "Fluorescence Lifetime Imaging (FLIM)", "microscopy"),
    ("fpm", "Fourier Ptychographic Microscopy", "microscopy"),
    ("palm_storm", "PALM/STORM Single-Molecule Localization", "microscopy"),
    ("polarization", "Polarization Microscopy", "microscopy"),
    # Electron microscopy
    ("sem", "Scanning Electron Microscopy (SEM)", "electron_microscopy"),
    ("tem", "Transmission Electron Microscopy (TEM)", "electron_microscopy"),
    ("stem", "Scanning TEM (STEM)", "electron_microscopy"),
    ("electron_tomography", "Electron Tomography", "electron_microscopy"),
    ("electron_diffraction", "4D-STEM Electron Diffraction", "electron_microscopy"),
    ("electron_holography", "Electron Holography", "electron_microscopy"),
    ("ebsd", "Electron Backscatter Diffraction (EBSD)", "electron_microscopy"),
    ("eels", "Electron Energy Loss Spectroscopy (EELS)", "electron_microscopy"),
    # Clinical optics
    ("oct", "Optical Coherence Tomography (OCT)", "clinical_optics"),
    ("octa", "OCT Angiography", "clinical_optics"),
    ("fundus", "Fundus Camera", "clinical_optics"),
    ("endoscopy", "Fiber Bundle Endoscopy", "clinical_optics"),
    # Computational
    ("light_field", "Light Field Imaging", "computational"),
    ("integral", "Integral Photography", "computational"),
    ("lensless", "Lensless (Diffuser Camera) Imaging", "computational_photography"),
    ("panorama", "Panorama Multi-Focus Fusion", "computational_photography"),
    # Neural rendering
    ("nerf", "Neural Radiance Fields (NeRF)", "neural_rendering"),
    ("gaussian_splatting", "3D Gaussian Splatting", "neural_rendering"),
    # Depth imaging
    ("tof_camera", "Time-of-Flight Depth Camera", "depth_imaging"),
    ("structured_light", "Structured-Light Depth Camera", "depth_imaging"),
    ("lidar", "LiDAR Scanner", "depth_imaging"),
    # Remote sensing
    ("sar", "Synthetic Aperture Radar (SAR)", "remote_sensing"),
    ("sonar", "Sonar Imaging", "remote_sensing"),
    # Particle imaging
    ("neutron_tomo", "Neutron Radiography / Tomography", "particle_imaging"),
    ("proton_radiography", "Proton Radiography", "particle_imaging"),
    ("muon_tomo", "Muon Tomography", "particle_imaging"),
]

_FALLBACK_MODALITIES = [
    SimpleNamespace(modality_key=k, display_name=d, category=c)
    for k, d, c in _FALLBACK_MODALITIES_RAW
]

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Pages"])

templates = Jinja2Templates(directory="pwm_platform/templates")


# ── Public pages (visible to everyone) ──────────────────────────────────


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login page — CompareGPT SSO redirect."""
    return templates.TemplateResponse("login.html", {
        "request": request,
        "sso_enabled": bool(settings.SSO_REDIRECT_URL),
        "sso_url": settings.SSO_REDIRECT_URL,
    })


@router.get("/sso/callback")
async def sso_callback(
    request: Request,
    response: Response,
    token: str = "",
    access_token: str = "",
    db: AsyncSession = Depends(get_db),
):
    """Handle SSO redirect callback — exchange token and set cookie."""
    sso_token = token or access_token
    if not sso_token:
        return RedirectResponse("/login?error=missing_token")

    try:
        result = await auth_service.exchange_sso_token(sso_token, db)
        redirect = RedirectResponse("/", status_code=302)
        redirect.set_cookie(
            key="access_token",
            value=result["access_token"],
            httponly=True,
            secure=True,
            samesite="lax",
            max_age=7 * 24 * 3600,
            path="/",
        )
        return redirect
    except Exception as exc:
        logger.error("SSO callback error: %s", exc)
        return RedirectResponse("/login?error=sso_failed")


@router.get("/", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Dashboard — public overview of platform stats and recent runs."""
    # Show all recent runs (public view)
    runs_result = await db.execute(
        select(Run).order_by(Run.submitted_at.desc()).limit(20)
    )
    runs = runs_result.scalars().all()

    count_result = await db.execute(select(func.count()).select_from(Run))
    total_runs = count_result.scalar() or 0

    modality_count = await db.execute(select(func.count()).select_from(ModalityBasics))
    total_modalities = modality_count.scalar() or 0

    dataset_count = await db.execute(select(func.count()).select_from(Dataset))
    total_datasets = dataset_count.scalar() or 0

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "user": user,
        "runs": runs,
        "total_runs": total_runs,
        "total_modalities": total_modalities,
        "total_datasets": total_datasets,
    })


@router.get("/runs/new", response_class=HTMLResponse)
async def new_run_page(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """New run form — requires login."""
    modalities_result = await db.execute(
        select(ModalityBasics).order_by(ModalityBasics.display_name)
    )
    modalities = modalities_result.scalars().all()

    # Fallback: if DB has no modalities seeded yet, use hardcoded list
    if not modalities:
        modalities = _FALLBACK_MODALITIES

    return templates.TemplateResponse("run_new.html", {
        "request": request,
        "user": user,
        "modalities": modalities,
    })


@router.get("/runs/{run_id}", response_class=HTMLResponse)
async def run_status_page(
    request: Request,
    run_id: str,
    user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Run status page — public view with live polling via HTMX."""
    result = await db.execute(select(Run).where(Run.run_id == run_id))
    run = result.scalar_one_or_none()
    if run is None:
        return templates.TemplateResponse("404.html", {
            "request": request, "user": user, "message": "Run not found"
        }, status_code=404)

    # Get triad report if available
    report = None
    if run.status == "completed":
        report_result = await db.execute(
            select(TriadReport).where(TriadReport.run_id == run_id)
        )
        report = report_result.scalar_one_or_none()

    return templates.TemplateResponse("run_status.html", {
        "request": request,
        "user": user,
        "run": run,
        "report": report,
    })


@router.get("/datasets", response_class=HTMLResponse)
async def datasets_page(
    request: Request,
    user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Dataset catalog — public."""
    result = await db.execute(
        select(Dataset).order_by(Dataset.created_at.desc()).limit(100)
    )
    datasets = result.scalars().all()

    return templates.TemplateResponse("datasets.html", {
        "request": request,
        "user": user,
        "datasets": datasets,
    })


@router.get("/modalities", response_class=HTMLResponse)
async def modalities_page(
    request: Request,
    category: str | None = None,
    user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Modality catalog — serves from Physics World Model knowledge base."""
    from pwm_platform.services.modality_database import (
        MODALITY_DATABASE,
        list_all_categories,
        list_all_modality_keys,
        list_modalities_by_category,
    )

    if category:
        keys = list_modalities_by_category(category)
    else:
        keys = list_all_modality_keys()

    # Build template-friendly objects with modality_key included
    modalities = []
    for k in keys:
        entry = dict(MODALITY_DATABASE[k])
        entry["modality_key"] = k
        modalities.append(entry)

    return templates.TemplateResponse("modalities.html", {
        "request": request,
        "user": user,
        "modalities": modalities,
        "categories": list_all_categories(),
        "selected_category": category,
    })


# ── Auth-required pages (PWM reconstruction actions) ────────────────────


@router.get("/bootstrap/new", response_class=HTMLResponse)
async def bootstrap_new_page(
    request: Request,
    user: User = Depends(get_current_user),
):
    """New modality bootstrap wizard — requires login."""
    return templates.TemplateResponse("bootstrap_new.html", {
        "request": request,
        "user": user,
    })


@router.get("/bootstrap/review", response_class=HTMLResponse)
async def bootstrap_review_page(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Bootstrap review queue (admin/reviewer) — requires login."""
    if user.role not in ("admin", "reviewer"):
        return RedirectResponse("/")

    result = await db.execute(
        select(BootstrapProposal)
        .where(BootstrapProposal.status.in_(["submitted", "under_review"]))
        .order_by(BootstrapProposal.submitted_at.desc())
    )
    proposals = result.scalars().all()

    return templates.TemplateResponse("bootstrap_review.html", {
        "request": request,
        "user": user,
        "proposals": proposals,
    })
