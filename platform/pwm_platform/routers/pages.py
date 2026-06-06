"""
Pages Router — server-rendered HTML pages (Jinja2 + HTMX).

Public pages: all viewing pages (SpecLab, datasets, modalities, run status).
Auth-required: actions that run PWM reconstruction (new run, bootstrap, review).
Login: CompareGPT SSO redirect flow.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from pwm_platform.auth.dependencies import get_current_user, get_optional_user
from pwm_platform.auth.service import auth_service
from pwm_platform.config import settings
from pwm_platform.db.database import get_db
from pwm_platform.db.models import (
    BootstrapProposal,
    ChallengeSubmission,
    ContributorProfile,
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


@router.get("/about", response_class=HTMLResponse)
async def about_page(request: Request, user: Optional[User] = Depends(get_optional_user)):
    """About page — introduction to Physics World Model as a Dyson Swarm."""
    return templates.TemplateResponse(request, "about.html", {"user": user, "active": "about"})


@router.get("/roadmap", response_class=HTMLResponse)
async def roadmap_page(request: Request, user: Optional[User] = Depends(get_optional_user)):
    """Roadmap — project phases and current status."""
    return templates.TemplateResponse(request, "roadmap.html", {"user": user, "active": "roadmap"})


@router.get("/specs", include_in_schema=False)
async def specs_page_legacy_redirect(request: Request):
    """Legacy URL — 301 to the renamed /digital-twins (preserve query string)."""
    qs = request.url.query
    return RedirectResponse(url="/digital-twins" + (f"?{qs}" if qs else ""), status_code=301)


@router.get("/digital-twins", response_class=HTMLResponse)
async def specs_page(
    request: Request,
    q: str = "",
    domain: str = "",
    status: str = "all",
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_optional_user),
):
    """L2 Digital Twins — browse all submitted digital twins."""
    from pwm_platform.db.models import PWMSubmission
    from sqlalchemy import desc, func, select

    query = (
        select(PWMSubmission)
        .where(PWMSubmission.submission_type == "spec")
        .order_by(desc(PWMSubmission.submitted_at))
    )
    if status != "all":
        query = query.where(PWMSubmission.status == status)
    result = await db.execute(query)
    all_specs = result.scalars().all()

    # Filter by search query
    if q:
        q_lower = q.lower()
        all_specs = [
            s for s in all_specs
            if q_lower in str(s.form_data).lower()
        ]

    # Stats
    counts_result = await db.execute(
        select(PWMSubmission.status, func.count(PWMSubmission.id))
        .where(PWMSubmission.submission_type == "spec")
        .group_by(PWMSubmission.status)
    )
    counts = dict(counts_result.all())

    return templates.TemplateResponse(request, "specs.html", {
        "user": user,
        "active": "specs",
        "specs": all_specs,
        "counts": counts,
        "total": sum(counts.values()),
        "q": q,
        "status_filter": status,
    })


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login page — SIWE wallet + Coinbase Wallet + Web3Auth + Google + email.

    Also pulls the on-chain config (Base Sepolia / Base mainnet) from the
    onchain router so Web3Auth creates wallets on the same chain the rest
    of the site operates on. Without this the embedded wallet would land
    on Base mainnet by default (Web3Auth's hardcoded fallback), and the
    user would need to manually switch chains before doing anything.
    """
    from pwm_platform.routers.onchain import _CONFIG as _ONCHAIN
    return templates.TemplateResponse(request, "login.html", {
        "google_client_id": settings.GOOGLE_CLIENT_ID,
        "sso_url": settings.SSO_REDIRECT_URL,
        "web3auth_client_id": settings.WEB3AUTH_CLIENT_ID,
        "web3auth_network": settings.WEB3AUTH_NETWORK,
        # Chain config matches what /api/v1/onchain/config exposes — same
        # chainId, rpcUrl, explorer for the embedded wallet.
        "chain_id_hex": "0x" + format(_ONCHAIN.get("chainId") or 84532, "x"),
        "chain_id_int": _ONCHAIN.get("chainId") or 84532,
        "chain_rpc_url": _ONCHAIN.get("rpcUrl") or "https://sepolia.base.org",
        "chain_explorer_url": _ONCHAIN.get("explorerUrl") or "https://sepolia.basescan.org",
        "chain_display_name": "Base Sepolia" if (_ONCHAIN.get("chainId") == 84532) else "Base",
    })


@router.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    """Sign-up page — create new account."""
    return templates.TemplateResponse(request, "signup.html", {
        "google_client_id": settings.GOOGLE_CLIENT_ID,
    })


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """User account settings — API key + PWM token balance."""
    from pwm_platform.services.pwm_token_service import pwm_token_service

    key = user.api_key or ""
    masked_key = (key[:8] + "..." + key[-4:]) if key else None

    pwm_balance_data = await pwm_token_service.get_balance(user.id, db)
    pwm_transactions = await pwm_token_service.list_transactions(user.id, db, limit=10)

    return templates.TemplateResponse(request, "settings.html", {
        "user": user,
        "has_api_key": bool(key),
        "masked_key": masked_key,
        "pwm_balance": pwm_balance_data["balance"],
        "pwm_lifetime_earned": pwm_balance_data["lifetime_earned"],
        "pwm_wallet": pwm_balance_data["on_chain_address"],
        "pwm_is_custodial": pwm_balance_data["is_custodial"],
        "pwm_transactions": pwm_transactions,
    })


@router.get("/forgot-password", response_class=HTMLResponse)
async def forgot_password_page(request: Request):
    """Forgot password page."""
    return templates.TemplateResponse(request, "forgot_password.html")


@router.get("/reset-password", response_class=HTMLResponse)
async def reset_password_page(request: Request, token: str = ""):
    """Password reset page (with token from email link)."""
    return templates.TemplateResponse(request, "reset_password.html", {
        "token": token,
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
        redirect = RedirectResponse("/benchmark", status_code=302)
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


_sidebar_cache: dict | None = None

# ── Carrier → noise model suggestions for fine-tuning prompts ────────────
_CARRIER_NOISE_SUGGESTIONS: dict[str, list[str]] = {
    "Photon": [
        "Switch to mixed Poisson-Gaussian noise with read noise σ=5 electrons",
        "Use pure Poisson shot noise for photon-limited regime",
    ],
    "X-ray": [
        "Switch to Poisson noise to model photon counting",
        "Add beam-hardening artifacts to the noise model",
    ],
    "Gamma": [
        "Set Poisson noise with mean count rate of 1000 counts/pixel",
        "Add scatter noise as a fraction of total counts",
    ],
    "Spin/RF": [
        "Set Gaussian noise with SNR=30 dB",
        "Add Rician noise to model MRI magnitude images",
    ],
    "RF": [
        "Use Gaussian noise with SNR=20 dB",
        "Add multiplicative speckle noise",
    ],
    "Electron": [
        "Set Poisson shot noise for low-dose imaging",
        "Add detector DQE degradation to the noise model",
    ],
    "Acoustic": [
        "Set Gaussian white noise with SNR=40 dB",
        "Add reverberant clutter noise",
    ],
    "Mechanical": [
        "Set Gaussian noise with thermal drift σ=0.1 nm",
        "Add 1/f noise for low-frequency scanning artifacts",
    ],
    "Neutron": [
        "Set Poisson noise for low-flux neutron beam",
    ],
}

# ── Category → forward-model refinement suggestions ──────────────────────
_CATEGORY_DAG_SUGGESTIONS: dict[str, list[str]] = {
    "medical": [
        "Add a projection primitive for multi-angle acquisition",
        "Insert a Fourier sampling step for k-space encoding",
    ],
    "microscopy": [
        "Add structured illumination before the PSF convolution",
        "Insert a wavelength selection primitive for multi-channel imaging",
    ],
    "compressive": [
        "Add a wavelength dispersion primitive for spectral coding",
        "Replace the random mask with a Hadamard pattern",
    ],
    "electron_microscopy": [
        "Add a CTF (contrast transfer function) convolution step",
        "Insert a projection primitive for tilt-series tomography",
    ],
    "remote_sensing": [
        "Add a Fourier sampling step for aperture synthesis",
        "Insert a motion/rotation primitive for multi-pass acquisition",
    ],
    "coherent": [
        "Add a second propagation branch for reference beam interference",
        "Insert a rotation primitive for tomographic phase retrieval",
    ],
    "scanning_probe": [
        "Add a convolution primitive for tip-sample interaction",
        "Insert a structured illumination step for multi-probe operation",
    ],
    "depth_imaging": [
        "Add a structured illumination primitive for coded patterns",
        "Insert a temporal modulation step for time-gating",
    ],
    "quantum": [
        "Add a summation primitive for coincidence counting",
        "Insert a second detection branch for correlation measurements",
    ],
    "ultrafast": [
        "Add temporal coding with a chirped mask",
        "Insert a rotation primitive for angular multiplexing",
    ],
    "spectroscopy": [
        "Add a wavelength dispersion step for spectral resolution",
        "Insert a Fourier sampling primitive for interferometric encoding",
    ],
    "astronomy": [
        "Add an atmospheric turbulence convolution step",
        "Insert a rotation primitive for Earth-rotation synthesis",
    ],
}


def _generate_finetune_examples(entry: dict) -> list[dict]:
    """Generate per-modality fine-tuning prompt examples.

    Each example is a dict with 'label' (short heading) and 'prompt' (full text).
    """
    import re as _re

    examples: list[dict] = []
    display_name = entry.get("display_name", "")
    mismatch_params = entry.get("mismatch_params", [])
    carrier = entry.get("carrier", "")
    category = entry.get("category", "")
    primitives = entry.get("primitives", [])
    spec_notation = entry.get("spec_notation", "")

    # 1. Mismatch parameter tuning (up to 2 examples from actual params)
    for p in mismatch_params[:2]:
        name = p.get("name", "")
        desc = p.get("description", name.replace("_", " "))
        nominal = p.get("nominal", 0)
        perturbed = p.get("perturbed", 0)
        # Clean human-readable name: strip trailing units like (px), (deg), (-)
        human_name = _re.sub(r"\s*\([^)]*\)\s*$", "", desc).strip().lower()
        if not human_name:
            human_name = name.replace("_", " ")
        if nominal == 0 and perturbed != 0:
            examples.append({
                "label": f"Tune {human_name}",
                "prompt": f"Set the {human_name} to {perturbed} and show how it affects reconstruction quality",
            })
        elif nominal != 0:
            # Suggest a different perturbation
            delta = abs(perturbed - nominal)
            new_val = nominal + 2 * delta if delta else nominal * 1.2
            new_val = round(new_val, 4)
            examples.append({
                "label": f"Tune {human_name}",
                "prompt": f"Change the {human_name} from {nominal} to {new_val} and explain the impact",
            })

    # 2. Add/remove mismatch param (if the modality has params, suggest adding another)
    if mismatch_params:
        existing_names = {p.get("name", "") for p in mismatch_params}
        # Suggest a param that's physically relevant but not already listed
        generic_suggestions = [
            ("detector_gain", "detector gain", 1.0, 1.05),
            ("noise_sigma", "noise level sigma", 0.01, 0.05),
            ("alignment_error", "alignment error in pixels", 0.0, 0.5),
            ("temperature_drift", "temperature drift", 0.0, 0.1),
        ]
        for pname, pdesc, pnom, ppert in generic_suggestions:
            if pname not in existing_names:
                examples.append({
                    "label": "Add mismatch param",
                    "prompt": f"Add a mismatch parameter for {pdesc} with nominal={pnom} and perturbed={ppert}",
                })
                break
    else:
        # No mismatch params — suggest adding one
        examples.append({
            "label": "Add mismatch param",
            "prompt": f"Add a mismatch parameter for detector gain with nominal=1.0 and perturbed=1.05",
        })

    # 3. Noise model change (carrier-specific)
    noise_suggestions = _CARRIER_NOISE_SUGGESTIONS.get(carrier, [])
    if noise_suggestions:
        examples.append({
            "label": "Change noise model",
            "prompt": noise_suggestions[0],
        })
    else:
        examples.append({
            "label": "Change noise model",
            "prompt": "Switch to Gaussian noise with SNR=25 dB",
        })

    # 4. Forward model modification (category-specific)
    dag_suggestions = _CATEGORY_DAG_SUGGESTIONS.get(category, [])
    if dag_suggestions:
        examples.append({
            "label": "Modify forward model",
            "prompt": dag_suggestions[0],
        })

    # 5. System variant / configuration change
    if len(primitives) >= 2:
        # Suggest simplifying or extending the DAG
        examples.append({
            "label": "Simplify pipeline",
            "prompt": f"Simplify the {spec_notation} pipeline by removing the last intermediate step and explain the tradeoff",
        })

    return examples[:5]  # cap at 5 per modality


def _get_system_catalog_stats() -> dict:
    """Load system catalog summary stats for the benchmark page."""
    import json as _json
    catalog_path = Path(__file__).resolve().parent.parent / "static" / "benchmark-data" / "system_catalog.json"
    try:
        with open(catalog_path) as f:
            catalog = _json.load(f)
    except Exception:
        return {"total": 0, "categories": 0, "algorithms": 0, "avg_psnr": 0}

    cats = set()
    total_algos = 0
    psnr_sum = 0
    for v in catalog.values():
        cats.add(v.get("category", ""))
        total_algos += v.get("num_algorithms_in_catalog", 0)
        psnr_sum += v.get("best_psnr_db", 0)

    # Top 5 most affordable feasible systems
    affordable = sorted(catalog.values(), key=lambda x: x.get("capital_cost_k_usd", 9999))[:5]
    top_psnr = sorted(catalog.values(), key=lambda x: x.get("best_psnr_db", 0), reverse=True)[:5]

    return {
        "total": len(catalog),
        "categories": len(cats),
        "algorithms": total_algos,
        "avg_psnr": round(psnr_sum / max(len(catalog), 1), 1),
        "affordable": [{"name": s.get("display_name", ""), "cost_k": s.get("capital_cost_k_usd", 0),
                        "id": s.get("id", "")} for s in affordable],
        "top_psnr": [{"name": s.get("display_name", ""), "psnr": s.get("best_psnr_db", 0),
                      "method": s.get("best_method", ""), "id": s.get("id", "")} for s in top_psnr],
    }


_grouped_variants_cache: dict | None = None


def _build_grouped_variants() -> dict[str, list[dict]]:
    """Build category → variants mapping for Common Mode modality picker.

    Cached after first call.
    """
    global _grouped_variants_cache
    if _grouped_variants_cache is not None:
        return _grouped_variants_cache

    from pwm_platform.services.benchmark_database import VARIANT_DATABASE

    groups: dict[str, list[dict]] = {}
    for vk, v in sorted(VARIANT_DATABASE.items()):
        cat = v.get("category", "other")
        groups.setdefault(cat, []).append({
            "variant_key": vk,
            "display_name": v.get("display_name", vk),
        })
    _grouped_variants_cache = groups
    return _grouped_variants_cache


def _build_sidebar_data() -> dict:
    """Build sidebar context: categories with modalities + primitives.

    Cached after first call (static data, never changes at runtime).
    """
    global _sidebar_cache
    if _sidebar_cache is not None:
        return _sidebar_cache

    from pwm_platform.services.benchmark_database._modality_catalog import (
        MODALITY_CATALOG,
        get_categories,
    )
    from pwm_platform.services.benchmark_database._primitives import SPEC_PRIMITIVES
    from pwm_platform.services.benchmark_database import VARIANT_DATABASE, list_all_variant_keys
    from pwm_platform.services.example_datasets import EXAMPLE_DATASETS

    raw_cats = get_categories()

    # Reverse mapping: variant_key → example dataset key, AND example_key → example_key
    # (modality catalog IDs may differ from variant_keys, e.g. "cassi" vs "cassi")
    _variant_to_example = {v["variant_key"]: k for k, v in EXAMPLE_DATASETS.items()}
    # Also map by the example key itself (e.g. "spc" → "spc", "cassi" → "cassi")
    for k in EXAMPLE_DATASETS:
        _variant_to_example.setdefault(k, k)

    # Build parent_modality → list of variant_keys for benchmark links
    _parent_to_variants: dict[str, list[dict]] = {}
    for vk in list_all_variant_keys():
        v = VARIANT_DATABASE[vk]
        pm = v.get("parent_modality", "")
        if pm:
            bms = v.get("benchmarks", [])
            challenge = next((b for b in bms if b.get("is_challenge")), None)
            # Get public + dev tier download paths
            tier_downloads = []
            if challenge:
                for tier_key, tier_label in [("public", "Public"), ("dev", "Dev")]:
                    tier_ds = challenge.get("tiers", {}).get(tier_key, {}).get("dataset", {})
                    gcs_path = tier_ds.get("gcs_object_path", "")
                    if gcs_path:
                        tier_downloads.append({
                            "tier": tier_label,
                            "url": f"/gcs/{gcs_path}",
                            "filename": gcs_path.rsplit("/", 1)[-1],
                        })
            _parent_to_variants.setdefault(pm, []).append({
                "variant_key": vk,
                "display_name": v.get("display_name", vk),
                "has_challenge": challenge is not None,
                "tier_downloads": tier_downloads,
            })

    # Build ordered dict with per-modality fine-tuning examples
    ordered_categories: dict[str, list[dict]] = {}
    for cat_slug in sorted(raw_cats.keys()):
        mod_ids = sorted(raw_cats[cat_slug])
        mods = []
        for mod_id in mod_ids:
            entry = MODALITY_CATALOG.get(mod_id, {})
            # Check if this modality has an example dataset
            example_key = _variant_to_example.get(mod_id)
            example_info = None
            if example_key:
                ex = EXAMPLE_DATASETS[example_key]
                example_info = {
                    "key": example_key,
                    "display_name": ex["display_name"],
                    "measurement_shape": ex["measurement_shape"],
                    "has_matrix": ex.get("has_matrix", False),
                    "matrix_shape": ex.get("matrix_shape", ""),
                    "has_gt": ex.get("has_gt", False),
                    "prompt_example": ex["prompt_example"],
                }
            # Benchmark variants for this modality
            benchmark_variants = _parent_to_variants.get(mod_id, [])
            mods.append({
                "id": mod_id,
                "display_name": entry.get("display_name", mod_id),
                "spec_notation": entry.get("spec_notation", ""),
                "carrier": entry.get("carrier", ""),
                "canonical_dag": entry.get("canonical_dag", ""),
                "finetune_examples": _generate_finetune_examples(entry),
                "example_dataset": example_info,
                "benchmark_variants": benchmark_variants,
            })
        ordered_categories[cat_slug] = mods

    total_count = sum(len(v) for v in ordered_categories.values())

    _sidebar_cache = {
        "sidebar_categories": ordered_categories,
        "sidebar_modality_count": total_count,
        "sidebar_primitives": SPEC_PRIMITIVES,
    }
    return _sidebar_cache


@router.get("/", response_class=HTMLResponse)
async def home_page(
    request: Request,
    user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Homepage — hero + flagship leaderboards + benchmark grid."""
    from collections import OrderedDict

    from pwm_platform.services.benchmark_database import (
        VARIANT_DATABASE,
        list_all_variant_keys,
    )

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
        ("scanning_probe", "Scanning Probe"),
        ("industrial_inspection", "Industrial Inspection"),
        ("spectroscopy", "Spectroscopy"),
        ("astronomy", "Astronomy"),
        ("ultrafast", "Ultrafast"),
        ("quantum", "Quantum"),
        ("experimental_science", "Experimental Science"),
        ("scientific_instrumentation", "Scientific Instrumentation"),
        ("multi_modal_fusion", "Multi-Modal Fusion"),
    ]
    category_labels = dict(CATEGORY_ORDER)

    grouped: dict[str, list[dict]] = OrderedDict()
    for cat_key, _label in CATEGORY_ORDER:
        grouped[cat_key] = []

    for key in list_all_variant_keys():
        entry = dict(VARIANT_DATABASE[key])
        entry["variant_key"] = key
        benchmarks = entry.get("benchmarks", [])
        entry["num_benchmarks"] = len(benchmarks)
        challenge = next((b for b in benchmarks if b.get("is_challenge")), None)
        entry["leaderboard"] = challenge.get("leaderboard", [])[:3] if challenge else []
        cat = entry.get("category", "other")
        if cat not in grouped:
            grouped[cat] = []
        grouped[cat].append(entry)

    grouped = OrderedDict((k, v) for k, v in grouped.items() if v)
    total_variants = sum(len(v) for v in grouped.values())

    FEATURED_KEYS = [
        "ct", "mri", "cryo_em", "ultrasound", "cassi",
        "nerf", "oct", "pet", "sar", "ghost_imaging",
    ]
    featured = []
    for fk in FEATURED_KEYS:
        entry = VARIANT_DATABASE.get(fk)
        if entry is None:
            continue
        normal_lb = entry.get("normal_leaderboard") or []
        featured.append({
            "variant_key": fk,
            "display_name": entry["display_name"],
            "category": category_labels.get(entry.get("category", ""), entry.get("category", "")),
            "leaderboard": normal_lb[:3],
        })

    return templates.TemplateResponse(request, "home.html", {
        "user": user,
        "grouped": grouped,
        "category_labels": category_labels,
        "total_variants": total_variants,
        "featured": featured,
        "active": "home",
    })


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_redirect():
    """Redirect old /dashboard URL to /ai4science."""
    return RedirectResponse("/ai4science", status_code=301)


# ── Pricing & Billing pages ──────────────────────────────────────────────


@router.get("/pricing", response_class=HTMLResponse)
async def pricing_page(
    request: Request,
    status: str = "",
    user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Subscription pricing page with Stripe + WeChat payment options."""
    from pwm_platform.services.billing_service import BillingService, PLANS, WECHAT_CREDIT_PACKS

    current_plan = "free"
    if user:
        svc = BillingService(db)
        balance = await svc.get_account_balance(user.id)
        current_plan = balance["plan_tier"]

    return templates.TemplateResponse(request, "pricing.html", {
        "user": user,
        "status": status,
        "current_plan": current_plan,
        "plans": PLANS,
        "wechat_packs": WECHAT_CREDIT_PACKS,
    })


@router.get("/subscription", response_class=HTMLResponse)
async def subscription_page(
    request: Request,
    status: str = "",
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """User subscription dashboard — credits, transactions, payments."""
    from pwm_platform.services.billing_service import BillingService

    svc = BillingService(db)
    balance = await svc.get_account_balance(user.id)
    plan_info = svc.get_plan_info(balance["plan_tier"])
    transactions = await svc.get_transaction_history(user.id, limit=20)
    payments = await svc.get_payment_history(user.id, limit=20)

    return templates.TemplateResponse(request, "subscription.html", {
        "user": user,
        "status": status,
        "balance": balance,
        "plan_features": plan_info.get("features", []),
        "transactions": transactions,
        "payments": payments,
    })


@router.get("/speclab", response_class=HTMLResponse)
async def speclab(
    request: Request,
    mode: str = "common",
    modality: str = "",
    algorithm: str = "",
    usage_type: str = "",
    type: str = "",  # alias from benchmark links (?type=reconstruct)
    user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """SpecLab — interactive spec builder with chat, simulation, and reconstruction."""
    # Visibility filter: logged-in users see public + own runs; anonymous see public only
    if user:
        visibility_filter = or_(Run.is_public == True, Run.user_id == user.id)  # noqa: E712
    else:
        visibility_filter = Run.is_public == True  # noqa: E712

    runs_result = await db.execute(
        select(Run).where(visibility_filter).order_by(Run.submitted_at.desc()).limit(20)
    )
    runs = runs_result.scalars().all()

    # Sidebar data: categories → modalities + primitives
    sidebar_data = _build_sidebar_data()

    # Chat history: fetch user's recent sessions for this variant (server-side rendering)
    chat_sessions: list[dict] = []
    if user:
        from pwm_platform.services.gemini_client import list_user_sessions
        chat_sessions = await list_user_sessions(db, user.id, variant_key="cassi")

    # Build grouped_variants for Common Mode modality picker
    grouped_variants = _build_grouped_variants()

    # Pre-populate algorithm list if modality is specified
    preselect_algorithms = []
    if modality:
        from pwm_platform.services.benchmark_database import get_algorithms, get_variant as _get_v

        v = _get_v(modality)
        if v:
            cat = v.get("category", "compressive")
            preselect_algorithms = get_algorithms(modality, cat)

    speclab_mode = mode if mode in ("common", "advanced") else "common"
    # Merge ?type= (from benchmark links) and ?usage_type= — default to reconstruct
    resolved_usage_type = usage_type or type or "reconstruct"

    return templates.TemplateResponse(request, "speclab.html", {
        "user": user,
        "runs": runs,
        "chat_variant_key": "cassi",
        "chat_sessions": chat_sessions,
        "sessions": chat_sessions,
        "current_session_id": "",
        "speclab_mode": speclab_mode,
        "grouped_variants": grouped_variants,
        "preselect_modality": modality,
        "preselect_algorithm": algorithm,
        "preselect_algorithms": preselect_algorithms,
        "usage_type": resolved_usage_type,
        **sidebar_data,
    }, headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"})


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

    return templates.TemplateResponse(request, "run_new.html", {
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
        return templates.TemplateResponse(request, "404.html", {
            "user": user, "message": "Run not found"
        }, status_code=404)

    # Access control: private runs only visible to owner or admin
    if not run.is_public:
        if user is None or (run.user_id != user.id and user.role != "admin"):
            return templates.TemplateResponse(request, "404.html", {
                "user": user, "message": "Run not found"
            }, status_code=404)

    is_owner = user is not None and run.user_id == user.id

    # Get triad report if available
    report = None
    if run.status == "completed":
        report_result = await db.execute(
            select(TriadReport).where(TriadReport.run_id == run_id)
        )
        report = report_result.scalar_one_or_none()

    return templates.TemplateResponse(request, "run_status.html", {
        "user": user,
        "run": run,
        "report": report,
        "is_owner": is_owner,
    })


@router.get("/benchmark", response_class=HTMLResponse)
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
        ("scanning_probe", "Scanning Probe"),
        ("industrial_inspection", "Industrial Inspection"),
        ("spectroscopy", "Spectroscopy"),
        ("astronomy", "Astronomy"),
        ("ultrafast", "Ultrafast"),
        ("quantum", "Quantum"),
        ("experimental_science", "Experimental Science"),
        ("scientific_instrumentation", "Scientific Instrumentation"),
        ("multi_modal_fusion", "Multi-Modal Fusion"),
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
        challenge = next((b for b in benchmarks if b.get("is_challenge")), None)
        entry["leaderboard"] = challenge.get("leaderboard", [])[:3] if challenge else []
        cat = entry.get("category", "other")
        if cat not in grouped:
            grouped[cat] = []
        grouped[cat].append(entry)

    # Remove empty categories
    grouped = OrderedDict((k, v) for k, v in grouped.items() if v)

    total_variants = sum(len(v) for v in grouped.values())

    # ── Featured Modalities — 10 attention-grabbing picks with mini leaderboards
    FEATURED_KEYS = [
        "ct", "mri", "cryo_em", "ultrasound", "cassi",
        "nerf", "oct", "pet", "sar", "ghost_imaging",
    ]
    featured = []
    for fk in FEATURED_KEYS:
        entry = VARIANT_DATABASE.get(fk)
        if entry is None:
            continue
        # Use standard (normal) leaderboard — shows PSNR/SSIM under ideal conditions
        normal_lb = entry.get("normal_leaderboard") or []
        lb = normal_lb[:3]
        featured.append({
            "variant_key": fk,
            "display_name": entry["display_name"],
            "category": category_labels.get(entry.get("category", ""), entry.get("category", "")),
            "leaderboard": lb,
        })

    # ── System Design Benchmark summary stats
    sys_catalog_stats = _get_system_catalog_stats()

    return templates.TemplateResponse(request, "datasets.html", {
        "user": user,
        "grouped": grouped,
        "category_labels": category_labels,
        "total_variants": total_variants,
        "featured": featured,
        "sys_catalog_stats": sys_catalog_stats,
    })


@router.get("/benchmark/contribution", response_class=HTMLResponse)
async def contribution_hub(
    request: Request,
    user: Optional[User] = Depends(get_optional_user),
):
    """Global contribution hub — 5 contribution types with credits/monetization."""
    return templates.TemplateResponse(request, "contribution_hub.html", {
        "user": user,
    })


@router.get("/benchmark/system-design")
async def system_design_redirect():
    """Redirect old /benchmark/system-design URL to /benchmark/contribution."""
    return RedirectResponse("/benchmark/contribution", status_code=301)


@router.get("/benchmark/system-design-legacy", response_class=HTMLResponse)
async def system_design_page(
    request: Request,
    user: Optional[User] = Depends(get_optional_user),
):
    """System Design Benchmark (PWM-SyS) — cross-modality system comparison."""
    import json as _json

    catalog_path = Path(__file__).resolve().parent.parent / "static" / "benchmark-data" / "system_catalog.json"
    try:
        with open(catalog_path) as f:
            catalog = _json.load(f)
    except Exception:
        catalog = {}

    # Group by category
    from collections import OrderedDict
    grouped: dict[str, list[dict]] = OrderedDict()
    for sys_id, sys_data in sorted(catalog.items(), key=lambda x: (x[1].get("category", ""), x[0])):
        cat = sys_data.get("category", "other")
        grouped.setdefault(cat, []).append(sys_data)

    return templates.TemplateResponse(request, "system_design.html", {
        "user": user,
        "catalog": catalog,
        "grouped": grouped,
        "total_systems": len(catalog),
        "total_categories": len(grouped),
    })


@router.get("/modalities", response_class=HTMLResponse)
async def modalities_page(
    request: Request,
    category: str | None = None,
    user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Modality catalog — serves all 169+ variants from benchmark database."""
    from pwm_platform.services.benchmark_database import VARIANT_DATABASE
    from pwm_platform.services.benchmark_database._modality_catalog import MODALITY_CATALOG
    from pwm_platform.services.modality_database import MODALITY_DATABASE

    # Build a merged catalog: VARIANT_DATABASE is authoritative for the full list;
    # MODALITY_DATABASE supplies rich physics metadata where available.
    all_entries: list[dict] = []
    seen: set[str] = set()
    for key, ventry in VARIANT_DATABASE.items():
        if key in seen:
            continue
        seen.add(key)
        entry: dict = {}
        # Rich metadata from MODALITY_DATABASE takes priority
        if key in MODALITY_DATABASE:
            entry = dict(MODALITY_DATABASE[key])
        else:
            # Supplement from MODALITY_CATALOG for modalities without a DB entry
            cat_entry = MODALITY_CATALOG.get(ventry.get("parent_modality", key), {})
            entry["display_name"] = ventry.get("display_name") or cat_entry.get("display_name") or key
            entry["category"] = ventry.get("category") or cat_entry.get("category") or "other"
            entry["description"] = cat_entry.get("description", "")
            entry["physics_class"] = cat_entry.get("carrier", "")
            entry["spec_notation"] = ventry.get("spec_notation") or cat_entry.get("spec_notation", "")
        entry["modality_key"] = key
        all_entries.append(entry)

    # Include MODALITY_DATABASE entries not covered by VARIANT_DATABASE (e.g. spc parent)
    for key, mentry in MODALITY_DATABASE.items():
        if key not in seen:
            entry = dict(mentry)
            entry["modality_key"] = key
            all_entries.append(entry)
            seen.add(key)

    # Sort alphabetically by display_name
    all_entries.sort(key=lambda e: e.get("display_name", "").lower())

    if category:
        all_entries = [e for e in all_entries if e.get("category") == category]

    # Collect all unique categories for the filter buttons
    all_categories = sorted({e.get("category", "") for e in all_entries if e.get("category")})

    return templates.TemplateResponse(request, "modalities.html", {
        "user": user,
        "modalities": all_entries,
        "categories": all_categories,
        "selected_category": category,
        "total_count": len(all_entries),
    })


@router.get("/modalities/{modality_key}", response_class=HTMLResponse)
async def modality_detail(
    request: Request,
    modality_key: str,
    user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Individual modality detail page with DAG viewer and trust badges."""
    from pwm_platform.services.benchmark_database import (
        get_variant,
        list_variants_for_modality,
        VARIANT_DATABASE,
    )
    from pwm_platform.services.benchmark_database._modality_catalog import MODALITY_CATALOG
    from pwm_platform.services.modality_database import MODALITY_DATABASE

    if modality_key not in MODALITY_DATABASE and modality_key not in VARIANT_DATABASE:
        raise HTTPException(404, f"Modality '{modality_key}' not found")

    if modality_key in MODALITY_DATABASE:
        modality = dict(MODALITY_DATABASE[modality_key])
    else:
        ventry = VARIANT_DATABASE[modality_key]
        parent = ventry.get("parent_modality", modality_key)
        cat_entry = MODALITY_CATALOG.get(parent, MODALITY_CATALOG.get(modality_key, {}))
        modality = {
            "display_name": ventry.get("display_name") or cat_entry.get("display_name") or modality_key,
            "category": ventry.get("category") or cat_entry.get("category") or "other",
            "description": cat_entry.get("description", ""),
            "physics_class": cat_entry.get("carrier", ""),
            "spec_notation": ventry.get("spec_notation") or cat_entry.get("spec_notation", ""),
            "spec_dag": ventry.get("spec_dag") or cat_entry.get("spec_dag", []),
        }

    # Find all benchmark variants for this modality
    variant_keys = list_variants_for_modality(modality_key)
    variants = []
    for vk in variant_keys:
        v = get_variant(vk)
        if v:
            v["variant_key"] = vk
            variants.append(v)

    # Compute SVG width for the DAG viewer
    spec_dag = modality.get("spec_dag") or []
    # Also try to pull spec_dag from the first variant if not on the modality itself
    if not spec_dag and variants:
        spec_dag = variants[0].get("spec_dag", [])
    dag_width = max(20 + len(spec_dag) * 160, 200) if spec_dag else 200

    # Spec notation: try modality first, fall back to first variant
    spec_notation = modality.get("spec_notation", "")
    if not spec_notation and variants:
        spec_notation = variants[0].get("spec_notation", "")

    # Related modalities in the same category — search VARIANT_DATABASE for breadth
    target_cat = modality.get("category", "")
    related_keys = [
        k for k, v in VARIANT_DATABASE.items()
        if v.get("category") == target_cat and k != modality_key
    ]
    related = []
    for rk in related_keys[:8]:
        rv = VARIANT_DATABASE[rk]
        rname = rv.get("display_name") or (MODALITY_DATABASE.get(rk, {}).get("display_name")) or rk
        related.append({"key": rk, "display_name": rname})

    # Fetch maintainers for this modality from the contributor profiles table
    from sqlalchemy.orm import selectinload
    maintainers = []
    try:
        result = await db.execute(
            select(ContributorProfile).options(selectinload(ContributorProfile.user))
        )
        for p in result.scalars().all():
            if modality_key in (p.maintained_modalities or []):
                uname = p.user.username if p.user else ""
                uemail = p.user.email if p.user else ""
                maintainers.append({
                    "user_id": p.user_id,
                    "username": uname,
                    "email": uemail,
                    "roles": p.roles or [],
                    "badges": p.badges or [],
                })
    except Exception:
        maintainers = []

    return templates.TemplateResponse(request, "modality_detail.html", {
        "user": user,
        "modality": modality,
        "modality_key": modality_key,
        "variants": variants,
        "spec_dag": spec_dag,
        "dag_width": dag_width,
        "spec_notation": spec_notation,
        "related": related,
        "maintainers": maintainers,
    })


# ── Trust-tier infrastructure: merge community submissions into leaderboards ──


async def _merge_community_submissions(
    variant: dict,
    variant_key: str,
    db: AsyncSession,
) -> None:
    """Merge approved community ChallengeSubmission rows into the variant's
    challenge leaderboard in-place.  Each approved submission carries its own
    trust_tier (draft / author_confirmed / reproduced / certified) which is
    rendered as a badge on the leaderboard table.

    Only submissions with status='approved' and matching variant_key are merged.
    Duplicate method names are skipped (synthetic baseline takes precedence).
    """
    # Find the challenge benchmark entry inside the variant dict
    challenge = None
    for bm in variant.get("benchmarks", []):
        if bm.get("is_challenge"):
            challenge = bm
            break
    if challenge is None:
        return

    leaderboard = challenge.get("leaderboard", [])

    # Query approved community submissions for this variant
    try:
        result = await db.execute(
            select(ChallengeSubmission).where(
                ChallengeSubmission.variant_key == variant_key,
                ChallengeSubmission.status == "approved",
            )
        )
        submissions = result.scalars().all()
    except Exception:
        # DB might not have the table yet (pre-migration); silently skip
        return

    if not submissions:
        return

    # Build set of existing method names to avoid duplicates
    existing_methods = {e.get("method", "").lower() for e in leaderboard}

    for sub in submissions:
        method_lower = (sub.method_name or "").lower()
        if method_lower in existing_methods:
            continue

        # Extract scores from the submission's scored JSON
        scores = sub.scores or {}
        pub = scores.get("public", {})
        dev = scores.get("dev", {})
        hid = scores.get("hidden", {})

        pub_score = pub.get("score")
        dev_score = dev.get("score")
        hid_score = hid.get("score")

        score_vals = [s for s in (pub_score, dev_score, hid_score) if s is not None]
        overall = round(sum(score_vals) / len(score_vals), 3) if score_vals else 0.0

        entry = {
            "rank": 0,
            "method": sub.method_name,
            "public_score": pub_score,
            "dev_score": dev_score,
            "hidden_score": hid_score,
            "overall_score": overall,
            "details": {
                "public": {
                    "psnr": pub.get("psnr"),
                    "ssim": pub.get("ssim"),
                    "consistency": pub.get("consistency"),
                },
                "dev": {
                    "psnr": dev.get("psnr"),
                    "ssim": dev.get("ssim"),
                    "consistency": dev.get("consistency"),
                },
                "hidden": {
                    "psnr": hid.get("psnr"),
                    "ssim": hid.get("ssim"),
                    "consistency": hid.get("consistency"),
                },
            },
            "source": f"Community — {sub.submitter.username}" if sub.submitter else "Community",
            "trust_tier": sub.trust_tier or "draft",
            "submission_id": sub.submission_id,
        }
        leaderboard.append(entry)
        existing_methods.add(method_lower)

    # Re-sort and re-rank entire leaderboard
    leaderboard.sort(key=lambda e: e.get("overall_score", 0), reverse=True)
    for i, e in enumerate(leaderboard, 1):
        e["rank"] = i
    challenge["leaderboard"] = leaderboard


def _merge_approved_claims_into_leaderboard(variant: dict, variant_key: str) -> None:
    """Merge approved claims from /tmp/pwm_claim_queue/ into the normal leaderboard.

    Approved claims appear at Draft trust tier alongside the catalog algorithms.
    This bridges the claims workflow with the benchmark leaderboard.
    """
    import json as _json

    claims_dir = Path("/tmp/pwm_claim_queue")
    if not claims_dir.exists():
        return

    normal_lb = variant.get("normal_leaderboard")
    if normal_lb is None:
        normal_lb = []
        variant["normal_leaderboard"] = normal_lb

    existing_methods = {e.get("method", "").lower() for e in normal_lb}

    for f in sorted(claims_dir.glob("*.json")):
        try:
            claim = _json.loads(f.read_text())
        except Exception:
            continue

        # Only approved claims for this modality
        if claim.get("status") != "approved":
            continue
        claim_mod = (claim.get("modality") or "").lower()
        if claim_mod != variant_key.lower() and claim_mod != variant.get("parent_modality", "").lower():
            continue

        method = claim.get("method") or claim.get("title", "")[:30]
        if method.lower() in existing_methods:
            continue

        # Parse PSNR/SSIM
        try:
            psnr = float(claim.get("claimed_psnr") or 0)
        except (ValueError, TypeError):
            psnr = 0.0
        try:
            ssim = float(claim.get("claimed_ssim") or 0)
        except (ValueError, TypeError):
            ssim = 0.0

        # Compute score same as catalog: 0.5 * clip((psnr-15)/30, 0, 1) + 0.5 * ssim
        psnr_norm = max(0.0, min(1.0, (psnr - 15.0) / 30.0))
        score = round(0.5 * psnr_norm + 0.5 * ssim, 4)

        entry = {
            "rank": 0,
            "method": method,
            "score": score,
            "psnr": round(psnr, 2),
            "ssim": round(ssim, 4),
            "source": claim.get("source_id") or "Claim",
            "trust_tier": claim.get("trust_tier", "draft"),
            "dataset": variant_key,
            "claim_id": claim.get("claim_id", ""),
        }
        normal_lb.append(entry)
        existing_methods.add(method.lower())

    # Re-sort and re-rank
    normal_lb.sort(key=lambda e: e.get("score", 0), reverse=True)
    for i, e in enumerate(normal_lb, 1):
        e["rank"] = i
    variant["normal_leaderboard"] = normal_lb


@router.get("/benchmark/{variant_key}", response_class=HTMLResponse)
async def variant_benchmarks_page(
    request: Request,
    variant_key: str,
    user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Variant benchmark page — benchmarks, modality intro, spec DAG, leaderboards, credits."""
    from pwm_platform.services.benchmark_database import (
        get_benchmark_gallery,
        get_spec_primitives,
        get_variant,
    )
    from pwm_platform.services.modality_database import MODALITY_DATABASE

    variant = get_variant(variant_key)
    if variant is None:
        return templates.TemplateResponse(request, "404.html", {
            "user": user, "message": "Variant not found"
        }, status_code=404)

    # Fetch parent modality data for the introduction section
    parent_key = variant.get("parent_modality")
    modality = None
    if parent_key and parent_key in MODALITY_DATABASE:
        modality = dict(MODALITY_DATABASE[parent_key])

    # Load pre-computed benchmark gallery (multi-scene scenario comparison)
    benchmark_gallery = get_benchmark_gallery(variant_key)

    # Set download URLs to the authenticated GCS proxy endpoint
    for bm in variant.get("benchmarks", []):
        pd = bm.get("public_dataset")
        if pd and pd.get("gcs_object_path"):
            pd["download_url"] = f"/gcs/{pd['gcs_object_path']}"

        # Wire challenge tier download URLs (Public + Dev)
        if bm.get("is_challenge"):
            for tier_key in ("public", "dev"):
                tier_ds = bm.get("tiers", {}).get(tier_key, {}).get("dataset")
                if tier_ds and tier_ds.get("gcs_object_path"):
                    tier_ds["download_url"] = f"/gcs/{tier_ds['gcs_object_path']}"

    # ── Merge approved community submissions into challenge leaderboards ──
    await _merge_community_submissions(variant, variant_key, db)

    # ── Merge approved claims into normal leaderboard ──
    _merge_approved_claims_into_leaderboard(variant, variant_key)

    # Map variant_key to InverseNet paper figure subdirectory
    _PAPER_FIG_MAP = {
        "cassi": "cassi",
        "cacti": "cacti",
        "spc_block": "spc",
        "spc_kronecker": "spc",
    }
    paper_fig_dir = _PAPER_FIG_MAP.get(variant_key, variant_key)

    # Build algorithm comparison gallery from algorithms/ subdirectory
    # Maps gallery_variant (e.g. spc_block) -> algorithms/scene_XX/recon_{key}.png
    algo_comparison = []
    gallery_variant = None
    for bm in variant.get("benchmarks", []):
        if bm.get("is_challenge"):
            gallery_variant = bm.get("gallery_variant", variant_key)
            break
    if gallery_variant is None:
        gallery_variant = variant_key

    algo_base = (
        Path(__file__).resolve().parent.parent
        / "static" / "img" / "benchmark_gallery" / gallery_variant / "algorithms"
    )
    if algo_base.is_dir():
        # Auto-detect algorithm keys from first scene
        for si in range(20):
            sd = algo_base / f"scene_{si:02d}"
            if not sd.is_dir() or not (sd / "gt.png").exists():
                break
            url_base = f"/static/img/benchmark_gallery/{gallery_variant}/algorithms/scene_{si:02d}"
            # Find all recon_*.png files
            recon_files = sorted(sd.glob("recon_*.png"))
            algos = []
            for rf in recon_files:
                key = rf.stem.replace("recon_", "")
                algos.append({"key": key, "name": key.replace("-", " ").replace("_", " ").title()})
            if algos:
                algo_comparison.append({
                    "scene_idx": si,
                    "base_url": url_base,
                    "algorithms": algos,
                })

    from pwm_platform.routers.redesign import MAINNET_VARIANTS

    return templates.TemplateResponse(request, "variant_benchmarks.html", {
        "user": user,
        "variant": variant,
        "variant_key": variant_key,
        "modality": modality,
        "primitives": get_spec_primitives(),
        "benchmark_gallery": benchmark_gallery,
        "paper_fig_dir": paper_fig_dir,
        "algo_comparison": algo_comparison,
        "on_mainnet": variant_key in MAINNET_VARIANTS,
    })


@router.get("/benchmark/{variant_key}/challenge/{tier_name}", response_class=HTMLResponse)
async def challenge_tier_page(
    request: Request,
    variant_key: str,
    tier_name: str,
    user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Challenge tier detail page — expanded view of a single tier."""
    from pwm_platform.services.benchmark_database import (
        get_benchmark_gallery,
        get_challenge_config,
        get_variant,
    )

    variant = get_variant(variant_key)
    if variant is None:
        return templates.TemplateResponse(request, "404.html", {
            "user": user, "message": "Variant not found"
        }, status_code=404)

    # Find the challenge benchmark
    challenge = None
    for bm in variant.get("benchmarks", []):
        if bm.get("is_challenge"):
            challenge = bm
            break

    if challenge is None:
        return templates.TemplateResponse(request, "404.html", {
            "user": user, "message": "No challenge benchmark for this variant"
        }, status_code=404)

    # Validate tier name
    if tier_name not in ("public", "dev", "hidden"):
        return templates.TemplateResponse(request, "404.html", {
            "user": user, "message": "Invalid tier name"
        }, status_code=404)

    tier = challenge.get("tiers", {}).get(tier_name)
    if tier is None:
        return templates.TemplateResponse(request, "404.html", {
            "user": user, "message": "Tier not found"
        }, status_code=404)

    # Wire download URLs for challenge tiers
    for tk in ("public", "dev"):
        tier_ds = challenge.get("tiers", {}).get(tk, {}).get("dataset")
        if tier_ds and tier_ds.get("gcs_object_path"):
            tier_ds["download_url"] = f"/gcs/{tier_ds['gcs_object_path']}"

    # Load benchmark gallery for scores (public tier only — dev/hidden use simulated data)
    benchmark_gallery = get_benchmark_gallery(variant_key) if tier_name == "public" else None

    # Map variant_key to InverseNet paper figure subdirectory
    _PAPER_FIG_MAP = {
        "cassi": "cassi",
        "cacti": "cacti",
        "spc_block": "spc",
        "spc_kronecker": "spc",
    }
    paper_fig_dir = _PAPER_FIG_MAP.get(variant_key, variant_key)

    # Per-modality extra chart filename (3rd figure below scenario_comparison)
    _EXTRA_CHART_MAP = {
        "cassi": "per_scene_psnr.png",
        "cacti": "per_video_psnr.png",
        "spc": "psnr_distribution.png",
    }
    paper_extra_chart = _EXTRA_CHART_MAP.get(paper_fig_dir)

    # ── Merge approved community submissions into challenge leaderboard ──
    await _merge_community_submissions(variant, variant_key, db)

    # Build per-tier leaderboard: filter + re-rank by this tier's score
    tier_leaderboard = []
    tier_score_key = f"{tier_name}_score"
    for entry in challenge.get("leaderboard", []):
        score = entry.get(tier_score_key)
        if score is not None:
            tier_leaderboard.append(entry)
    tier_leaderboard.sort(key=lambda e: e.get(tier_score_key, 0), reverse=True)

    # Data preview images for challenge tier pages
    # Multi-view labels for hand-crafted variants with spectral/temporal views
    _MULTI_VIEW_LABELS = {
        "cassi": ("Band 7 (~450 nm)", "Band 21 (~650 nm)"),
        "cacti": ("Frame 0 (t=0)", "Frame 7 (t=7)"),
    }
    _TIER_SCENE_SHARED = {"public": 0, "dev": 1, "hidden": 2}

    # Auto-detect preview images from gallery directory
    # Try tier-specific gallery first (e.g. spc_block/dev/scene_00), fall back to shared
    gallery_key = challenge.get("gallery_variant", variant_key)
    gallery_base = (
        Path(__file__).resolve().parent.parent
        / "static" / "img" / "benchmark_gallery" / gallery_key
    )
    # Prefer tier-specific directory — dev uses scene_02 (best representative), public uses scene_00
    _TIER_PREVIEW_SCENE = {"public": 0, "dev": 2}
    tier_preview_idx = _TIER_PREVIEW_SCENE.get(tier_name, 0)
    tier_gallery_dir = gallery_base / tier_name / f"scene_{tier_preview_idx:02d}"
    scene_idx_shared = _TIER_SCENE_SHARED.get(tier_name, 0)
    shared_gallery_dir = gallery_base / f"scene_{scene_idx_shared:02d}"
    if tier_gallery_dir.is_dir() and (tier_gallery_dir / "gt.png").exists():
        scene_idx = tier_preview_idx
        gallery_dir = tier_gallery_dir
        base = f"/static/img/benchmark_gallery/{gallery_key}/{tier_name}/scene_{scene_idx:02d}"
    else:
        scene_idx = scene_idx_shared
        gallery_dir = shared_gallery_dir
        base = f"/static/img/benchmark_gallery/{gallery_key}/scene_{scene_idx:02d}"
    # Determine recon base URL: if tier-specific dir lacks recon images, fall back to shared
    recon_base = base
    if gallery_dir != shared_gallery_dir:
        has_recon = any(gallery_dir.glob("recon_*.png"))
        if not has_recon:
            # Fall back to shared scene directory for recon images
            shared_scene_dir = gallery_base / f"scene_{scene_idx:02d}"
            if shared_scene_dir.is_dir() and any(shared_scene_dir.glob("recon_*.png")):
                recon_base = f"/static/img/benchmark_gallery/{gallery_key}/scene_{scene_idx:02d}"
    # Only show data preview for public and dev tiers (hidden has no visible data)
    data_preview = None
    if tier_name in ("public", "dev") and gallery_dir.is_dir() and (gallery_dir / "gt.png").exists():
        if (gallery_dir / "gt_view1.png").exists():
            labels = _MULTI_VIEW_LABELS.get(gallery_key, ("View 1", "View 2"))
            data_preview = {
                "is_multi": True,
                "view1_label": labels[0],
                "view2_label": labels[1],
                "scene_idx": scene_idx,
                "base_url": base,
                "recon_base_url": recon_base,
            }
        else:
            data_preview = {
                "is_multi": False,
                "scene_idx": scene_idx,
                "base_url": base,
                "recon_base_url": recon_base,
            }
        # Check for best-algorithm reconstruction image
        # Try tier-specific dir first (algorithms_dev/), then shared (algorithms/)
        _BEST_RECON_PRIORITY = [
            ("recon_lpd.png", "Learned Primal-Dual + gradient"),
            ("recon_dolce.png", "DOLCE + gradient"),
            ("recon_pnp-drunet.png", "PnP-DRUNet + blind cal"),
            ("recon_fista-tv-tuned.png", "FISTA-TV + blind cal"),
        ]
        for _algo_dir_name in (f"algorithms_{tier_name}", "algorithms"):
            algo_scene_dir = gallery_base / _algo_dir_name / f"scene_{scene_idx:02d}"
            if not algo_scene_dir.is_dir():
                continue
            for recon_file, recon_label in _BEST_RECON_PRIORITY:
                if (algo_scene_dir / recon_file).exists():
                    _url_algo_dir = _algo_dir_name
                    data_preview["best_recon_url"] = (
                        f"/static/img/benchmark_gallery/{gallery_key}"
                        f"/{_url_algo_dir}/scene_{scene_idx:02d}/{recon_file}"
                    )
                    data_preview["best_recon_label"] = recon_label
                    break
            if "best_recon_url" in data_preview:
                break

    # Extract true_spec only if the tier's visible_data includes it
    tier_true_spec = None
    if "true_spec" in tier.get("visible_data", []):
        tier_true_spec = tier.get("true_spec")

    # Build algorithm comparison demo: auto-detect recon_*.png in gallery
    # Only for public tier (dev has no visible ground truth for meaningful comparison)
    algo_demo = []
    if tier_name == "public":
        algo_dir_name = "algorithms"
        algo_base = gallery_base / algo_dir_name
        for si in range(20):
            sd = algo_base / f"scene_{si:02d}"
            if not sd.is_dir() or not (sd / "gt.png").exists():
                break
            recon_files = sorted(sd.glob("recon_*.png"))
            if not recon_files:
                continue
            _ALGO_DISPLAY = {
                # CT / sinogram
                "fbp": "FBP", "fbpconv": "FBPConvNet", "lpd": "Learned Primal-Dual",
                "dolce": "DOLCE", "pnp-admm": "PnP-ADMM", "pnp-drunet": "PnP-DRUNet",
                "tv-admm": "TV-ADMM", "red-cnn": "RED-CNN", "dudotrans": "DuDoTrans",
                "piner-ct": "PINER-CT", "ct-fm": "CT-FM",
                # MRI / k-space
                "zero-filled-ifft": "Zero-Filled IFFT", "sense": "SENSE",
                "grappa": "GRAPPA", "l1-wavelet": "L1-Wavelet",
                "kt-sparse-sense": "k-t SPARSE-SENSE", "espirit": "ESPIRiT",
                "loraks": "LORAKS", "bm3d-mri": "BM3D-MRI",
                "modl": "MoDL", "hybridcascade": "HybridCascade",
                "e2e-varnet": "E2E-VarNet", "swinmr": "SwinMR",
                "humus-net": "HUMUS-Net", "humus-net-pp": "HUMUS-Net++",
                "reconformer": "ReconFormer",
                # Denoising / microscopy
                "tv-denoising": "TV-Denoising", "nlm": "NLM",
                "nlm-tv": "NLM+TV", "bm3d": "BM3D",
                "richardson-lucy": "Richardson-Lucy", "wiener": "Wiener Filter",
                "dncnn": "DnCNN", "ffdnet": "FFDNet", "drunet": "DRUNet",
                "restormer": "Restormer", "uformer": "Uformer",
                # Phase retrieval
                "angular-spectrum": "Angular Spectrum", "gerchberg-saxton": "Gerchberg-Saxton",
                "hio": "HIO", "prdeep": "prDeep",
                # General
                "tikhonov": "Tikhonov", "admm": "ADMM", "ista": "ISTA", "fista": "FISTA",
                "red": "RED", "pnp-hqs": "PnP-HQS", "score-mri": "Score-MRI",
                # CT variants
                "fbp-tv": "FBP+TV", "sart": "SART", "osem": "OSEM",
                "art-tv": "ART-TV", "cgls": "CGLS", "bm3d-ct": "BM3D-CT",
                # Ultrasound
                "das": "Delay-and-Sum", "universal-back-proj": "Universal Back-Proj",
                # Radio/astronomy
                "clean": "CLEAN", "matched-filter": "Matched Filter",
                # Phase
                "angular-spectrum": "Angular Spectrum", "gerchberg-saxton": "Gerchberg-Saxton",
                "tv-phase": "TV (Phase)", "hio": "HIO",
                # Nuclear
                "mlem": "MLEM", "osem-pet": "OSEM-PET",
                # Misc
                "phase-unwrap": "Phase Unwrap", "l-bfgs-fwi": "L-BFGS FWI",
                "bilateral": "Bilateral Filter", "tv-strong": "TV (strong)",
                "nlm-tv": "NLM+TV", "bilateral-tv": "Bilateral+TV",
                "espirit": "ESPIRiT",
            }
            algos = []
            for rf in recon_files:
                key = rf.stem.replace("recon_", "")
                name = _ALGO_DISPLAY.get(key, key.replace("-", " ").replace("_", " ").title())
                algos.append({"key": key, "name": name})
            url_algo_dir = algo_dir_name if (gallery_base / algo_dir_name / f"scene_{si:02d}").is_dir() else "algorithms"
            algo_demo.append({
                "scene_idx": si,
                "base_url": f"/static/img/benchmark_gallery/{gallery_key}/{url_algo_dir}/scene_{si:02d}",
                "algorithms": algos,
            })

    # Extract Scenario I baselines for this tier (measured results from actual HDF5)
    baselines = challenge.get("baselines", {})
    scenario_i_baselines = [
        b for b in baselines.get("scenario_i", [])
        if b.get("tier", "public") == tier_name
    ]
    # Scenario II/III (published literature) — show on public tier only
    scenario_ii_baselines = baselines.get("scenario_ii", []) if tier_name == "public" else []
    scenario_iii_baselines = baselines.get("scenario_iii", []) if tier_name == "public" else []

    return templates.TemplateResponse(request, "challenge_tier.html", {
        "user": user,
        "variant": variant,
        "variant_key": variant_key,
        "challenge": challenge,
        "tier_name": tier_name,
        "tier": tier,
        "benchmark_gallery": benchmark_gallery,
        "tier_leaderboard": tier_leaderboard,
        "paper_fig_dir": paper_fig_dir,
        "paper_extra_chart": paper_extra_chart,
        "data_preview": data_preview,
        "recon_gallery": [],
        "algo_demo": algo_demo,
        "spec_ranges": tier.get("spec_ranges", challenge.get("spec_ranges", [])),
        "true_spec": tier_true_spec,
        "scenario_i_baselines": scenario_i_baselines,
        "scenario_ii_baselines": scenario_ii_baselines,
        "scenario_iii_baselines": scenario_iii_baselines,
    })


@router.get("/benchmark/{variant_key}/compete", response_class=HTMLResponse)
async def compete_page(
    request: Request,
    variant_key: str,
    user: Optional[User] = Depends(get_optional_user),
):
    """Competition info page — public, sign-in only needed for submissions."""
    from pwm_platform.services.benchmark_database import get_variant

    variant = get_variant(variant_key)
    if variant is None:
        return templates.TemplateResponse(request, "404.html", {
            "user": user, "message": "Variant not found"
        }, status_code=404)

    challenge = None
    for bm in variant.get("benchmarks", []):
        if bm.get("is_challenge"):
            challenge = bm
            break

    # Wire download URLs for challenge tiers
    if challenge:
        for tk in ("public", "dev"):
            tier_ds = challenge.get("tiers", {}).get(tk, {}).get("dataset")
            if tier_ds and tier_ds.get("gcs_object_path"):
                tier_ds["download_url"] = f"/gcs/{tier_ds['gcs_object_path']}"

    return templates.TemplateResponse(request, "compete.html", {
        "user": user,
        "variant": variant,
        "variant_key": variant_key,
        "challenge": challenge,
    })


@router.get("/benchmark/{variant_key}/contribute", response_class=HTMLResponse)
async def contribute_page(
    request: Request,
    variant_key: str,
    user: Optional[User] = Depends(get_optional_user),
):
    """Contribution info page — public, sign-in only needed for submissions."""
    from pwm_platform.services.benchmark_database import get_variant

    variant = get_variant(variant_key)
    if variant is None:
        return templates.TemplateResponse(request, "404.html", {
            "user": user, "message": "Variant not found"
        }, status_code=404)

    return templates.TemplateResponse(request, "contribute.html", {
        "user": user,
        "variant": variant,
        "variant_key": variant_key,
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

    return templates.TemplateResponse(request, "my_runs.html", {
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
    return templates.TemplateResponse(request, "bootstrap_new.html", {
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
        return RedirectResponse("/benchmark")

    result = await db.execute(
        select(BootstrapProposal)
        .where(BootstrapProposal.status.in_(["submitted", "under_review"]))
        .order_by(BootstrapProposal.submitted_at.desc())
    )
    proposals = result.scalars().all()

    return templates.TemplateResponse(request, "bootstrap_review.html", {
        "user": user,
        "proposals": proposals,
    })


@router.get("/submissions/review", response_class=HTMLResponse)
async def submissions_review_page(
    request: Request,
    status: str | None = None,
    category: str | None = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Challenge submissions review queue (admin/reviewer) — requires login."""
    from sqlalchemy import func

    if user.role not in ("admin", "reviewer"):
        return RedirectResponse("/benchmark")

    stmt = select(ChallengeSubmission).order_by(ChallengeSubmission.submitted_at.desc())
    if status in ("pending", "approved", "rejected"):
        stmt = stmt.where(ChallengeSubmission.status == status)
    if category in ("competition", "contribution"):
        stmt = stmt.where(ChallengeSubmission.category == category)

    result = await db.execute(stmt)
    submissions = result.scalars().all()

    # Counts for filter tabs
    count_result = await db.execute(
        select(ChallengeSubmission.category, func.count())
        .group_by(ChallengeSubmission.category)
    )
    counts = dict(count_result.all())

    return templates.TemplateResponse(request, "submissions_review.html", {
        "user": user,
        "submissions": submissions,
        "status_filter": status,
        "category_filter": category,
        "competition_count": counts.get("competition", 0),
        "contribution_count": counts.get("contribution", 0),
    })


@router.get("/claims", response_class=HTMLResponse)
async def claims_review_page(
    request: Request,
    user: Optional[User] = Depends(get_optional_user),
    status: str = "all",
):
    """Claim review queue page with optional status filter."""
    import json as _json

    claims_dir = Path("/tmp/pwm_claim_queue")
    claims = []
    if claims_dir.exists():
        for f in sorted(claims_dir.glob("*.json"), reverse=True):
            try:
                c = _json.loads(f.read_text())
                if status != "all" and c.get("status") != status:
                    continue
                claims.append(c)
            except Exception:
                continue
    return templates.TemplateResponse(
        request, "claim_review.html",
        {"user": user, "claims": claims, "status_filter": status},
    )


@router.get("/datasets", response_class=HTMLResponse)
async def datasets_browser_page(
    request: Request,
    user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Dataset browser page — browse, filter, and register datasets."""
    import json as _json

    datasets: list[dict] = []

    # 1. Load from database (Dataset model)
    try:
        result = await db.execute(select(Dataset).order_by(Dataset.created_at.desc()))
        for d in result.scalars().all():
            datasets.append({
                "modality": d.modality,
                "name": d.dataset_id,
                "description": d.description or "",
                "num_samples": d.num_samples,
                "format": (d.tags or [{}])[0] if isinstance(d.tags, list) and d.tags else "",
                "license": d.license or "internal",
                "source_url": d.source or "",
            })
    except Exception:
        pass

    # 2. Load from JSON file store (/tmp/pwm_datasets/)
    ds_dir = Path("/tmp/pwm_datasets")
    if ds_dir.exists():
        for f in sorted(ds_dir.glob("*.json")):
            try:
                entry = _json.loads(f.read_text())
                datasets.append({
                    "modality": entry.get("modality", ""),
                    "name": entry.get("name", f.stem),
                    "description": entry.get("description", ""),
                    "num_samples": entry.get("num_samples"),
                    "format": entry.get("format", ""),
                    "license": entry.get("license", "internal"),
                    "source_url": entry.get("source_url", ""),
                })
            except Exception:
                continue

    modalities = sorted(set(d["modality"] for d in datasets if d["modality"]))

    return templates.TemplateResponse(request, "datasets_browser.html", {
        "user": user,
        "datasets": datasets,
        "modalities": modalities,
        "selected_modality": "",
    })


@router.get("/instruments", response_class=HTMLResponse)
async def instruments_page(
    request: Request,
    user: Optional[User] = Depends(get_optional_user),
):
    """Instrument registry page — browse registered InstrumentCards."""
    import json as _json

    instruments: list[dict] = []

    # Load from /tmp/pwm_ct_qc_demo/ (pre-existing demo data)
    demo_card = Path("/tmp/pwm_ct_qc_demo/instrument_card.json")
    demo_qc = Path("/tmp/pwm_ct_qc_demo/ct_qc_diagnostic_report.json")
    if demo_card.exists():
        try:
            card = _json.loads(demo_card.read_text())
            qc_report = None
            if demo_qc.exists():
                qc_report = _json.loads(demo_qc.read_text())
            card["qc_report"] = qc_report
            instruments.append(card)
        except Exception:
            pass

    # Load from /tmp/pwm_instruments/ (user-registered)
    inst_dir = Path("/tmp/pwm_instruments")
    if inst_dir.exists():
        for f in sorted(inst_dir.glob("*.json")):
            try:
                card = _json.loads(f.read_text())
                # Avoid duplicates with demo card
                if card.get("instrument_id") != "siemens_somatom_001":
                    instruments.append(card)
            except Exception:
                continue

    return templates.TemplateResponse(request, "instruments.html", {
        "user": user,
        "instruments": instruments,
    })


@router.get("/instruments/{instrument_id}", response_class=HTMLResponse)
async def instrument_detail_page(
    request: Request,
    instrument_id: str,
    user: Optional[User] = Depends(get_optional_user),
):
    """Instrument detail page — drift charts, calibration history, QC metrics."""
    import json as _json

    card = None

    # Check demo card first
    demo_card = Path("/tmp/pwm_ct_qc_demo/instrument_card.json")
    if demo_card.exists():
        try:
            c = _json.loads(demo_card.read_text())
            if c.get("instrument_id") == instrument_id:
                card = c
        except Exception:
            pass

    # Check user-registered instruments
    if card is None:
        inst_file = Path(f"/tmp/pwm_instruments/{instrument_id}.json")
        if inst_file.exists():
            try:
                card = _json.loads(inst_file.read_text())
            except Exception:
                pass

    if card is None:
        raise HTTPException(status_code=404, detail=f"Instrument {instrument_id} not found")

    # Normalize file-store fields to match template expectations
    instrument = {
        "instrument_id": card.get("instrument_id", instrument_id),
        "name": card.get("name") or f"{card.get('manufacturer','')} {card.get('model','')}".strip() or instrument_id,
        "modality": card.get("modality", ""),
        "manufacturer": card.get("manufacturer", ""),
        "model_number": card.get("model", card.get("model_number", "")),
        "serial_number": card.get("serial_number", ""),
        "lab": card.get("lab", ""),
        "institution": card.get("institution", ""),
        "description": card.get("description", ""),
        "calibration_date": card.get("calibration_date") or card.get("last_qc_date"),
        "drift_budget_pct": card.get("drift_budget_pct", 0.0),
        "contact_email": card.get("contact_email", ""),
        "is_public": card.get("is_public", True),
        "card_data": card,
    }

    return templates.TemplateResponse(request, "instrument_detail.html", {
        "user": user,
        "instrument": instrument,
    })


@router.get("/reproduce", response_class=HTMLResponse)
async def reproduce_page(
    request: Request,
    user: Optional[User] = Depends(get_optional_user),
):
    """Reproduction queue — approved claims that need independent reproduction."""
    import json as _json

    claims_dir = Path("/tmp/pwm_claim_queue")
    claims: list[dict] = []
    modalities_set: set[str] = set()

    if claims_dir.exists():
        for f in sorted(claims_dir.glob("*.json"), reverse=True):
            try:
                c = _json.loads(f.read_text())
                # Only show approved claims at draft or author_confirmed tier
                if (c.get("status") == "approved"
                        and c.get("trust_tier") in ("draft", "author_confirmed")):
                    claims.append(c)
                    if c.get("modality"):
                        modalities_set.add(c["modality"])
            except Exception:
                continue

    return templates.TemplateResponse(request, "reproduce.html", {
        "user": user,
        "claims": claims,
        "modalities": sorted(modalities_set),
    })


@router.get("/admin/roles", response_class=HTMLResponse)
async def admin_roles_page(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Admin role management page -- assign roles and modality maintainership."""
    if user.role not in ("admin", "reviewer"):
        raise HTTPException(status_code=403, detail="Admin access required")

    from pwm_platform.routers.contributors import BADGE_DEFINITIONS, VALID_ROLES

    # Fetch all users with their contributor profiles
    users_result = await db.execute(
        select(User).order_by(User.id.asc()).limit(200)
    )
    all_users = users_result.scalars().all()

    # Fetch all contributor profiles
    profiles_result = await db.execute(select(ContributorProfile))
    profiles_list = profiles_result.scalars().all()
    profiles_by_user = {p.user_id: p for p in profiles_list}

    return templates.TemplateResponse(
        request,
        "admin_roles.html",
        {
            "user": user,
            "all_users": all_users,
            "profiles_by_user": profiles_by_user,
            "valid_roles": VALID_ROLES,
            "badge_definitions": BADGE_DEFINITIONS,
        },
    )


@router.get("/admin/users", response_class=HTMLResponse)
async def admin_users_page(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Admin user management page."""
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    result = await db.execute(select(User).order_by(User.id))
    users = result.scalars().all()
    return templates.TemplateResponse(
        request,
        "admin_users.html",
        {
            "user": user,
            "users": users,
        },
    )


@router.get("/admin/audit", response_class=HTMLResponse)
async def admin_audit_page(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Admin audit log page -- shows recent contribution history entries."""
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    from sqlalchemy.orm import selectinload as _sil

    result = await db.execute(
        select(ContributorProfile)
        .options(_sil(ContributorProfile.user))
        .order_by(ContributorProfile.id.desc())
    )
    profiles = result.scalars().all()

    # Collect all contribution_history entries
    audit_entries = []
    for p in profiles:
        username = p.user.username if p.user else f"User #{p.user_id}"
        for entry in (p.contribution_history or [])[-20:]:
            audit_entries.append({
                "user_id": p.user_id,
                "username": username,
                **entry,
            })

    # Sort by timestamp descending
    audit_entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    return templates.TemplateResponse(
        request,
        "admin_audit.html",
        {
            "user": user,
            "entries": audit_entries[:100],
        },
    )


# ── Solver Gallery, Gate Dashboard, Red Team pages ─────────────────────


# Tier classification for solver types
_TRADITIONAL_TYPES = frozenset({
    "Classical", "Variational", "PnP", "Compressed Sensing",
    "Low-Rank", "Statistical", "Dictionary Learning", "Self-Supervised",
})
_BEST_QUALITY_TYPES = frozenset({
    "Diffusion", "Diffusion Model", "Score-Based", "Score-based",
    "Foundation Model", "Physics-Informed", "Generative",
})


def _classify_tier(algo_type: str) -> str:
    """Classify an algorithm type into one of the three display tiers."""
    if algo_type in _TRADITIONAL_TYPES:
        return "traditional_cpu"
    if algo_type in _BEST_QUALITY_TYPES:
        return "best_quality"
    return "famous_dl"


def _build_solver_list() -> dict:
    """Build a deduplicated, tier-grouped solver list from the algorithm catalog."""
    from pwm_platform.services.benchmark_database._algorithm_catalog import (
        _CATEGORY_ALGORITHMS,
        _VARIANT_OVERRIDES,
    )

    # Collect (solver_name, type) -> {solver_info, modalities set}
    seen: dict[tuple[str, str], dict] = {}

    def _ingest(algos: list[dict], modality_key: str):
        for algo in algos:
            key = (algo["name"], algo.get("type", ""))
            if key not in seen:
                seen[key] = {
                    "name": algo["name"],
                    "type": algo.get("type", "Unknown"),
                    "params": algo.get("params", "0"),
                    "source": algo.get("source", ""),
                    "mask_aware": algo.get("mask_aware", False),
                    "available": algo.get("available", True),
                    "modalities": set(),
                }
            seen[key]["modalities"].add(modality_key)

    # Ingest variant overrides
    for variant_key, algos in _VARIANT_OVERRIDES.items():
        _ingest(algos, variant_key)

    # Ingest category algorithms
    for cat_key, algos in _CATEGORY_ALGORITHMS.items():
        _ingest(algos, cat_key)

    # Group by tier
    tiers: dict[str, list[dict]] = {
        "traditional_cpu": [],
        "famous_dl": [],
        "best_quality": [],
    }
    modality_keys: set[str] = set()
    for solver in seen.values():
        solver["modalities"] = sorted(solver["modalities"])
        modality_keys.update(solver["modalities"])
        tier = _classify_tier(solver["type"])
        tiers[tier].append(solver)

    # Sort each tier by name
    for tier_list in tiers.values():
        tier_list.sort(key=lambda s: s["name"].lower())

    return {
        "solvers_by_tier": tiers,
        "tier_counts": {k: len(v) for k, v in tiers.items()},
        "total_solvers": sum(len(v) for v in tiers.values()),
        "modality_keys": sorted(modality_keys),
        "modality_count": len(modality_keys),
    }


@router.get("/solvers", response_class=HTMLResponse)
async def solvers_page(
    request: Request,
    user=Depends(get_optional_user),
):
    """Solver Gallery — all available solvers grouped by tier."""
    import json as _json
    data = _build_solver_list()
    # Load community-registered solvers
    community: list[dict] = []
    solvers_dir = Path("/tmp/pwm_solvers")
    if solvers_dir.exists():
        for f in sorted(solvers_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                community.append(_json.loads(f.read_text()))
            except Exception:
                pass
    return templates.TemplateResponse(
        request,
        "solvers.html",
        {"user": user, "community_solvers": community, **data},
    )


@router.get("/api/v1/solvers")
async def list_solvers():
    """Return all available solvers from the algorithm catalog."""
    try:
        from pwm_platform.services.benchmark_database._algorithm_catalog import (
            _CATEGORY_ALGORITHMS,
            _VARIANT_OVERRIDES,
        )
    except ImportError:
        return {"solvers": [], "total": 0}

    solvers: dict[str, dict] = {}

    # Collect from category pools
    for category, algos in _CATEGORY_ALGORITHMS.items():
        for algo in algos:
            name = algo.get("name", "")
            if name and name not in solvers:
                solvers[name] = {
                    "name": name,
                    "type": algo.get("type", ""),
                    "params": algo.get("params", ""),
                    "source": algo.get("source", ""),
                    "category": category,
                }

    # Collect from variant overrides
    for variant_key, algos in _VARIANT_OVERRIDES.items():
        for algo in algos:
            name = algo.get("name", "")
            if name and name not in solvers:
                solvers[name] = {
                    "name": name,
                    "type": algo.get("type", ""),
                    "params": algo.get("params", ""),
                    "source": algo.get("source", ""),
                    "category": variant_key,
                }

    # Merge community-registered solvers
    import json as _json
    community_dir = Path("/tmp/pwm_solvers")
    if community_dir.exists():
        for f in sorted(community_dir.glob("*.json")):
            try:
                card = _json.loads(f.read_text())
                name = card.get("name", "")
                if name and name not in solvers:
                    solvers[name] = {
                        "name": name,
                        "type": card.get("solver_type", ""),
                        "params": card.get("params", ""),
                        "source": card.get("code_url", "community"),
                        "category": card.get("modality", "community"),
                        "solver_id": card.get("solver_id"),
                        "registered_by": card.get("registered_by"),
                    }
            except Exception:
                pass

    return {"solvers": list(solvers.values()), "total": len(solvers)}


@router.get("/gates", response_class=HTMLResponse)
async def gates_page(
    request: Request,
    user=Depends(get_optional_user),
):
    """Gate Dashboard — all active gates organized by layer."""
    import json as _json
    gates = {
        "operational": [
            {"id": "R1", "name": "Spec Completeness", "scope": "Universal", "checks": "All CoreSpec fields bound and type-valid"},
            {"id": "R2", "name": "Reproducibility", "scope": "Universal", "checks": "Seeds, versions, hashes recorded"},
            {"id": "R3", "name": "Metric Integrity", "scope": "Universal", "checks": "SHA-256 hashes match stored artifacts"},
            {"id": "R4", "name": "Budget Compliance", "scope": "Universal", "checks": "Runtime within declared budget"},
        ],
        "scientific": [
            {"id": "S1", "name": "Finite Specifiability", "scope": "Design time", "checks": "Problem admits finite type-valid description"},
            {"id": "S2", "name": "Hadamard Stability", "scope": "Design time", "checks": "Well-posed (existence, uniqueness, continuity)"},
            {"id": "S3", "name": "Approximability", "scope": "Design time", "checks": "Convergent discretization exists"},
            {"id": "S4", "name": "Certifiability", "scope": "Audit time", "checks": "Computable error bounds exist"},
        ],
        "imaging": [
            {"id": "G1", "name": "Recoverability", "scope": "Imaging", "checks": "Information-theoretic recovery limit"},
            {"id": "G2", "name": "Carrier Budget", "scope": "Imaging", "checks": "Measurement capacity (photon count, k-space)"},
            {"id": "G3", "name": "Operator Mismatch", "scope": "Imaging", "checks": "Forward model accuracy"},
        ],
        "ct_qc": [
            {"id": "drift", "name": "Drift Detection", "scope": "CT QC", "checks": "Temporal drift from baseline"},
            {"id": "artifact", "name": "Artifact Detection", "scope": "CT QC", "checks": "Ring, cupping, streak artifacts"},
            {"id": "threshold", "name": "Threshold Breach", "scope": "CT QC", "checks": "QC metric exceeds clinical limit"},
        ],
    }
    proposals = []
    proposals_dir = Path("/tmp/pwm_gate_proposals")
    if proposals_dir.exists():
        for p in sorted(proposals_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
            try:
                proposals.append(_json.loads(p.read_text()))
            except Exception:
                pass
    return templates.TemplateResponse(
        request,
        "gates.html",
        {"user": user, "gates": gates, "proposals": proposals},
    )


@router.post("/gates/propose")
async def gates_propose(
    request: Request,
    user=Depends(get_optional_user),
):
    """Submit a gate RFC proposal (JSON or form)."""
    import json as _json
    import datetime as _dt
    import secrets as _secrets

    if not user:
        raise HTTPException(status_code=401, detail="Login required to propose a gate RFC")

    content_type = request.headers.get("content-type", "")
    if "form" in content_type:
        form = await request.form()
        gate_name = form.get("gate_name", "")
        description = form.get("description", "")
        modality = form.get("modality", "")
        rationale = form.get("rationale", "")
    else:
        body = await request.json()
        gate_name = body.get("gate_name", "")
        description = body.get("description", "")
        modality = body.get("modality", "")
        rationale = body.get("rationale", "")

    if not gate_name or not description:
        raise HTTPException(status_code=400, detail="gate_name and description are required")

    proposal_id = f"rfc_{_secrets.token_hex(6)}_{_dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%d%H%M%S')}"
    proposal = {
        "proposal_id": proposal_id,
        "gate_name": gate_name,
        "description": description,
        "modality": modality,
        "rationale": rationale,
        "proposed_by": user.username if user else "anonymous",
        "status": "draft",
        "submitted_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }

    gates_dir = Path("/tmp/pwm_gate_proposals")
    gates_dir.mkdir(parents=True, exist_ok=True)
    (gates_dir / f"{proposal_id}.json").write_text(_json.dumps(proposal, indent=2))

    return proposal


@router.post("/gates/{proposal_id}/vote")
async def gates_vote(
    proposal_id: str,
    request: Request,
    user=Depends(get_optional_user),
):
    """Vote on a gate RFC proposal (admin/reviewer only).

    Body JSON: {"action": "approve"|"reject"|"comment", "comment": "..."}
    Three approvals with zero rejections → auto-accept.
    Any rejection → status stays draft until resolved.
    """
    import json as _json
    import datetime as _dt

    if not user:
        raise HTTPException(status_code=401, detail="Login required")
    if user.role not in ("admin", "reviewer"):
        raise HTTPException(status_code=403, detail="Only admin/reviewer can vote on RFCs")

    gates_dir = Path("/tmp/pwm_gate_proposals")
    proposal_path = gates_dir / f"{proposal_id}.json"
    if not proposal_path.exists():
        raise HTTPException(status_code=404, detail=f"Proposal {proposal_id} not found")

    body = await request.json()
    action = body.get("action", "")
    comment = body.get("comment", "")

    if action not in ("approve", "reject", "comment"):
        raise HTTPException(status_code=400, detail="action must be approve, reject, or comment")

    proposal = _json.loads(proposal_path.read_text())
    votes = proposal.setdefault("votes", [])

    # Prevent duplicate votes from same user (same action)
    existing = [v for v in votes if v["user"] == user.username and v["action"] == action]
    if existing and action != "comment":
        raise HTTPException(status_code=409, detail=f"You already voted '{action}' on this proposal")

    votes.append({
        "user": user.username,
        "role": user.role,
        "action": action,
        "comment": comment,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    })

    # Auto-accept: ≥3 approvals and 0 rejections
    n_approve = sum(1 for v in votes if v["action"] == "approve")
    n_reject = sum(1 for v in votes if v["action"] == "reject")

    if n_reject > 0:
        proposal["status"] = "disputed"
    elif n_approve >= 3:
        proposal["status"] = "accepted"
    elif n_approve >= 1:
        proposal["status"] = "under_review"

    proposal["votes"] = votes
    proposal_path.write_text(_json.dumps(proposal, indent=2))

    return {
        "proposal_id": proposal_id,
        "status": proposal["status"],
        "approvals": n_approve,
        "rejections": n_reject,
        "total_votes": len([v for v in votes if v["action"] != "comment"]),
    }


@router.get("/redteam", response_class=HTMLResponse)
async def redteam_page(
    request: Request,
    user=Depends(get_optional_user),
):
    """Red Team Dashboard — adversarial testing bounty board and reports."""
    import json as _json

    claims_dir = Path("/tmp/pwm_claim_queue")
    redteam_claims: list[dict] = []
    if claims_dir.exists():
        for f in claims_dir.glob("*.json"):
            try:
                c = _json.loads(f.read_text())
                if "[RED-TEAM]" in c.get("title", "") or c.get("method", "") == "red_team":
                    redteam_claims.append(c)
            except Exception:
                pass

    bounties = [
        {"gate": "R1", "challenge": "Create a manifest that passes R1 with missing CoreSpec fields", "difficulty": "Medium", "status": "Open"},
        {"gate": "R2", "challenge": "Produce non-reproducible results that still pass R2", "difficulty": "Hard", "status": "Open"},
        {"gate": "R3", "challenge": "Bypass SHA-256 hash verification", "difficulty": "Very Hard", "status": "Open"},
        {"gate": "R4", "challenge": "Hide budget overrun from R4 detection", "difficulty": "Medium", "status": "Open"},
        {"gate": "Social", "challenge": "Inflate metrics while passing all automated gates", "difficulty": "Easy (by design)", "status": "Documented"},
    ]

    return templates.TemplateResponse(
        request,
        "redteam.html",
        {"user": user, "claims": redteam_claims, "bounties": bounties},
    )
