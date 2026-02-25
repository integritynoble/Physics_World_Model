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
from sqlalchemy import func, or_, select
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
    """Dashboard — shows public runs + own runs for logged-in users."""
    from pwm_platform.services.modality_database import MODALITY_DATABASE

    # Visibility filter: logged-in users see public + own runs; anonymous see public only
    if user:
        visibility_filter = or_(Run.is_public == True, Run.user_id == user.id)  # noqa: E712
    else:
        visibility_filter = Run.is_public == True  # noqa: E712

    runs_result = await db.execute(
        select(Run).where(visibility_filter).order_by(Run.submitted_at.desc()).limit(20)
    )
    runs = runs_result.scalars().all()

    count_result = await db.execute(
        select(func.count()).select_from(Run).where(visibility_filter)
    )
    total_runs = count_result.scalar() or 0

    total_modalities = len(MODALITY_DATABASE)

    # Count unique canonical datasets across all modalities
    all_datasets = set()
    for entry in MODALITY_DATABASE.values():
        for ds in entry.get("canonical_datasets", []):
            all_datasets.add(ds)
    total_datasets = len(all_datasets)

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "user": user,
        "runs": runs,
        "total_runs": total_runs,
        "total_modalities": total_modalities,
        "total_datasets": total_datasets,
        "chat_variant_key": "sd_cassi",
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
    """Run status page — public runs visible to all, private runs only to owner/admin."""
    result = await db.execute(select(Run).where(Run.run_id == run_id))
    run = result.scalar_one_or_none()
    if run is None:
        return templates.TemplateResponse("404.html", {
            "request": request, "user": user, "message": "Run not found"
        }, status_code=404)

    # Access control: private runs only visible to owner or admin
    if not run.is_public:
        if user is None or (run.user_id != user.id and user.role != "admin"):
            return templates.TemplateResponse("404.html", {
                "request": request, "user": user, "message": "Run not found"
            }, status_code=404)

    is_owner = user is not None and run.user_id == user.id

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
        "is_owner": is_owner,
    })


@router.get("/datasets", response_class=HTMLResponse)
async def datasets_page(
    request: Request,
    user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Benchmark datasets — lists all variant benchmarks grouped by category."""
    from collections import OrderedDict

    from pwm_platform.services.benchmark_database import (
        VARIANT_DATABASE,
        list_all_variant_keys,
    )

    # Category display order and labels
    CATEGORY_ORDER = [
        ("compressive", "Compressive"),
        ("medical", "Medical"),
        ("medical_ultrasound", "Medical Ultrasound"),
        ("coherent", "Coherent"),
        ("microscopy", "Microscopy"),
        ("electron_microscopy", "Electron Microscopy"),
        ("clinical_optics", "Clinical Optics"),
        ("computational", "Computational"),
        ("computational_photography", "Computational Photography"),
        ("neural_rendering", "Neural Rendering"),
        ("depth_imaging", "Depth Imaging"),
        ("remote_sensing", "Remote Sensing"),
        ("particle_imaging", "Particle Imaging"),
    ]
    category_labels = dict(CATEGORY_ORDER)

    # Group variants by category
    grouped: dict[str, list[dict]] = OrderedDict()
    for cat_key, _label in CATEGORY_ORDER:
        grouped[cat_key] = []

    for key in list_all_variant_keys():
        entry = dict(VARIANT_DATABASE[key])
        entry["variant_key"] = key
        benchmarks = entry.get("benchmarks", [])
        entry["num_benchmarks"] = len(benchmarks)
        entry["num_public"] = sum(1 for b in benchmarks if b.get("has_public_dataset"))
        entry["num_hidden"] = sum(1 for b in benchmarks if b.get("has_hidden_dataset"))
        cat = entry.get("category", "other")
        if cat not in grouped:
            grouped[cat] = []
        grouped[cat].append(entry)

    # Remove empty categories
    grouped = OrderedDict((k, v) for k, v in grouped.items() if v)

    total_variants = sum(len(v) for v in grouped.values())

    return templates.TemplateResponse("datasets.html", {
        "request": request,
        "user": user,
        "grouped": grouped,
        "category_labels": category_labels,
        "total_variants": total_variants,
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


@router.get("/datasets/{variant_key}", response_class=HTMLResponse)
async def variant_benchmarks_page(
    request: Request,
    variant_key: str,
    user: Optional[User] = Depends(get_optional_user),
):
    """Variant benchmark page — benchmarks, modality intro, spec DAG, leaderboards, credits."""
    from pwm_platform.services.benchmark_database import (
        get_spec_primitives,
        get_variant,
    )
    from pwm_platform.services.modality_database import MODALITY_DATABASE

    variant = get_variant(variant_key)
    if variant is None:
        return templates.TemplateResponse("404.html", {
            "request": request, "user": user, "message": "Variant not found"
        }, status_code=404)

    # Fetch parent modality data for the introduction section
    parent_key = variant.get("parent_modality")
    modality = None
    if parent_key and parent_key in MODALITY_DATABASE:
        modality = dict(MODALITY_DATABASE[parent_key])

    return templates.TemplateResponse("variant_benchmarks.html", {
        "request": request,
        "user": user,
        "variant": variant,
        "variant_key": variant_key,
        "modality": modality,
        "primitives": get_spec_primitives(),
    })


@router.get("/my-runs", response_class=HTMLResponse)
async def my_runs_page(
    request: Request,
    visibility: str | None = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Personal runs page — shows all runs owned by the current user."""
    stmt = select(Run).where(Run.user_id == user.id).order_by(Run.submitted_at.desc())
    if visibility == "public":
        stmt = stmt.where(Run.is_public == True)  # noqa: E712
    elif visibility == "private":
        stmt = stmt.where(Run.is_public == False)  # noqa: E712

    runs_result = await db.execute(stmt)
    runs = runs_result.scalars().all()

    return templates.TemplateResponse("my_runs.html", {
        "request": request,
        "user": user,
        "runs": runs,
        "selected_visibility": visibility,
    })


# ── Auth-required pages (PWM reconstruction actions) ────────────────────


@router.get("/bootstrap/new", response_class=HTMLResponse)
async def bootstrap_new_page(
    request: Request,
    user: Optional[User] = Depends(get_optional_user),
):
    """New modality bootstrap wizard — public page."""
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
