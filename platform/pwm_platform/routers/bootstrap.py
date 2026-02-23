"""
Bootstrap Router — new modality bootstrap proposals with similarity engine.

When a user describes a new imaging modality, the bootstrap engine:
  1. Finds the top-3 most similar existing modalities (by matching physics_class,
     forward_model_family, sensor_type, source_type, geometry, noise_model)
  2. Generates starter templates (operator graph, experiment spec) from the
     best-matching modality
  3. Produces calibration checklists, simulation plans, and recommended metrics
  4. Returns an HTML partial that renders directly in the HTMX target
"""

from __future__ import annotations

import uuid
from collections import Counter

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from pwm_platform.auth.dependencies import get_current_user, get_optional_user, require_role
from pwm_platform.db.database import get_db
from pwm_platform.db.models import BootstrapProposal, User
from pwm_platform.services.modality_database import MODALITY_DATABASE

router = APIRouter(prefix="/api/v1/bootstrap", tags=["Bootstrap"])
templates = Jinja2Templates(directory="pwm_platform/templates")

# ── Similarity Engine ───────────────────────────────────────────────────────

# Fields used for matching, with weights
_MATCH_FIELDS = {
    "physics_class": 3,
    "forward_model_family": 3,
    "sensor_type": 2,
    "source_type": 2,
    "geometry": 2,
    "noise_model": 1,
    "wave_model": 1,
}


def _compute_similarity(query: dict, candidate: dict) -> float:
    """Compute weighted similarity score between query and a modality entry."""
    score = 0.0
    max_score = 0.0
    for field, weight in _MATCH_FIELDS.items():
        max_score += weight
        q_val = (query.get(field) or "").lower().strip()
        c_val = (candidate.get(field) or "").lower().strip()
        if not q_val:
            continue
        if q_val == c_val:
            score += weight
        elif q_val in c_val or c_val in q_val:
            score += weight * 0.5
    return score / max_score if max_score > 0 else 0.0


def find_similar_modalities(
    physics_class: str = "",
    forward_model_family: str = "",
    sensor_type: str = "",
    source_type: str = "",
    geometry: str = "",
    noise_model: str = "",
    top_k: int = 5,
) -> list[dict]:
    """Find the top-k most similar modalities from the knowledge base."""
    query = {
        "physics_class": physics_class,
        "forward_model_family": forward_model_family,
        "sensor_type": sensor_type,
        "source_type": source_type,
        "geometry": geometry,
        "noise_model": noise_model,
    }

    scored = []
    for key, entry in MODALITY_DATABASE.items():
        sim = _compute_similarity(query, entry)
        scored.append((sim, key, entry))

    scored.sort(key=lambda x: -x[0])
    results = []
    for sim, key, entry in scored[:top_k]:
        results.append({
            "modality_key": key,
            "display_name": entry["display_name"],
            "category": entry["category"],
            "physics_class": entry.get("physics_class", ""),
            "forward_model_family": entry.get("forward_model_family", ""),
            "sensor_type": entry.get("sensor_type", ""),
            "noise_model": entry.get("noise_model", ""),
            "default_solver": entry.get("default_solver", ""),
            "description": entry.get("description", "")[:200] + "...",
            "similarity": round(sim * 100, 1),
            "setup_diagram_url": entry.get("setup_diagram_url", ""),
            "recon_results": entry.get("recon_results"),
        })
    return results


def generate_bootstrap_templates(query: dict, best_match: dict) -> dict:
    """Generate starter templates from the best-matching modality."""
    key = best_match["modality_key"]
    entry = MODALITY_DATABASE[key]

    # Operator graph template — from best match
    op_graph = {
        "based_on": key,
        "forward_model_family": entry.get("forward_model_family", ""),
        "typical_x_dims": entry.get("typical_x_dims", [256, 256]),
        "typical_y_dims": entry.get("typical_y_dims", [256, 256]),
        "noise_model": query.get("noise_model") or entry.get("noise_model", "gaussian"),
        "note": f"Adapted from {entry['display_name']}. Adjust dimensions and parameters for your modality.",
    }

    # Experiment spec template
    exp_spec = {
        "based_on": key,
        "experimental_setup": entry.get("experimental_setup", {}),
        "note": "Modify these default parameters for your specific hardware.",
    }

    # Calibration checklist
    calibration = entry.get("calibration_params", [])
    if not calibration:
        calibration = ["system_response_function", "noise_characterization", "geometric_calibration"]

    # Simulation plan
    sim_plan = {
        "steps": [
            f"1. Define forward model using {entry.get('forward_model_family', 'linear')} family",
            f"2. Generate synthetic ground truth (typical dims: {entry.get('typical_x_dims', [256,256])})",
            f"3. Apply forward operator with {query.get('noise_model') or entry.get('noise_model', 'gaussian')} noise",
            f"4. Reconstruct using {entry.get('default_solver', 'gradient_descent')} solver",
            "5. Evaluate with PSNR, SSIM, NRMSE metrics",
            "6. Run W2 mismatch diagnosis",
        ],
        "based_on": key,
    }

    # Recommended metrics
    metrics = entry.get("evaluation_metrics", ["psnr", "ssim", "nrmse"])

    # Mismatch modes to watch for
    mismatch_modes = entry.get("mismatch_modes", [])

    # Reconstruction algorithms
    algorithms = []
    intro = entry.get("introduction", {})
    if intro and intro.get("common_algorithms"):
        algorithms = intro["common_algorithms"]

    # Common mistakes to avoid
    mistakes = []
    if intro and intro.get("common_mistakes"):
        mistakes = intro["common_mistakes"]

    return {
        "operator_graph": op_graph,
        "experiment_spec": exp_spec,
        "calibration_checklist": calibration,
        "simulation_plan": sim_plan,
        "recommended_metrics": metrics,
        "mismatch_modes": mismatch_modes,
        "recommended_algorithms": algorithms,
        "common_mistakes": mistakes,
    }


# ── Schemas ──────────────────────────────────────────────────────────────


class BootstrapCreateRequest(BaseModel):
    modality_key: str = Field(..., min_length=2)
    display_name: str = ""
    physics_class: str = ""
    forward_model_family: str = ""
    sensor_type: str = ""
    source_type: str = ""
    geometry: str = ""
    noise_model: str = "gaussian"


class BootstrapReviewRequest(BaseModel):
    decision: str = Field(..., pattern="^(approve|revise|reject)$")
    notes: str = ""


# ── Endpoints ────────────────────────────────────────────────────────────


@router.post("", response_class=HTMLResponse)
async def create_proposal(
    request: Request,
    modality_key: str = Form(...),
    display_name: str = Form(""),
    physics_class: str = Form(""),
    forward_model_family: str = Form(""),
    sensor_type: str = Form(""),
    source_type: str = Form(""),
    geometry: str = Form(""),
    noise_model: str = Form("gaussian"),
):
    """Submit a new modality bootstrap proposal.

    Accepts form data (from HTMX) and returns an HTML partial.
    No authentication required — results are ephemeral.
    """
    query = {
        "modality_key": modality_key,
        "display_name": display_name or modality_key,
        "physics_class": physics_class,
        "forward_model_family": forward_model_family,
        "sensor_type": sensor_type,
        "source_type": source_type,
        "geometry": geometry,
        "noise_model": noise_model,
    }

    # Check if modality already exists
    existing = modality_key in MODALITY_DATABASE
    if existing:
        entry = MODALITY_DATABASE[modality_key]
        return templates.TemplateResponse("_bootstrap_result.html", {
            "request": request,
            "already_exists": True,
            "modality_key": modality_key,
            "entry": entry,
        })

    # Find similar modalities
    similar = find_similar_modalities(
        physics_class=physics_class,
        forward_model_family=forward_model_family,
        sensor_type=sensor_type,
        source_type=source_type,
        geometry=geometry,
        noise_model=noise_model,
    )

    # Generate templates from best match
    bootstrap_templates = {}
    if similar:
        bootstrap_templates = generate_bootstrap_templates(query, similar[0])

    return templates.TemplateResponse("_bootstrap_result.html", {
        "request": request,
        "already_exists": False,
        "query": query,
        "similar": similar,
        "templates": bootstrap_templates,
    })


@router.get("")
async def list_proposals(
    status: str | None = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List bootstrap proposals."""
    stmt = select(BootstrapProposal).order_by(BootstrapProposal.submitted_at.desc())
    if status:
        stmt = stmt.where(BootstrapProposal.status == status)

    result = await db.execute(stmt)
    proposals = result.scalars().all()

    return {
        "proposals": [
            {
                "proposal_id": p.proposal_id,
                "modality_key": p.modality_key,
                "display_name": p.display_name,
                "status": p.status,
                "physics_class": p.physics_class,
                "submitted_at": p.submitted_at.isoformat() if p.submitted_at else None,
            }
            for p in proposals
        ],
        "total": len(proposals),
    }


@router.get("/{proposal_id}")
async def get_proposal(
    proposal_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get full proposal details."""
    result = await db.execute(
        select(BootstrapProposal).where(BootstrapProposal.proposal_id == proposal_id)
    )
    p = result.scalar_one_or_none()
    if p is None:
        raise HTTPException(status_code=404, detail="Proposal not found")

    return {
        "proposal_id": p.proposal_id,
        "modality_key": p.modality_key,
        "display_name": p.display_name,
        "status": p.status,
        "physics_class": p.physics_class,
        "forward_model_family": p.forward_model_family,
        "sensor_type": p.sensor_type,
        "source_type": p.source_type,
        "geometry": p.geometry,
        "noise_model": p.noise_model,
        "operator_graph_template": p.operator_graph_template,
        "experiment_spec_template": p.experiment_spec_template,
        "simulation_plan": p.simulation_plan,
        "collection_checklist": p.collection_checklist,
        "calibration_modes": p.calibration_modes,
        "recommended_metrics": p.recommended_metrics,
        "uncertainty_notes": p.uncertainty_notes,
        "viability_checklist": p.viability_checklist,
        "similar_modalities": p.similar_modalities,
        "review_notes": p.review_notes,
        "review_history": p.review_history,
    }


@router.patch("/{proposal_id}/review")
async def review_proposal(
    proposal_id: str,
    body: BootstrapReviewRequest,
    user: User = Depends(require_role("admin", "reviewer")),
    db: AsyncSession = Depends(get_db),
):
    """Review a bootstrap proposal (admin/reviewer only)."""
    result = await db.execute(
        select(BootstrapProposal).where(BootstrapProposal.proposal_id == proposal_id)
    )
    p = result.scalar_one_or_none()
    if p is None:
        raise HTTPException(status_code=404, detail="Proposal not found")

    status_map = {
        "approve": "approved",
        "revise": "revision_requested",
        "reject": "rejected",
    }
    p.status = status_map[body.decision]
    p.reviewer_id = user.id
    p.review_notes = body.notes

    history = list(p.review_history or [])
    history.append({
        "decision": body.decision,
        "reviewer_id": user.id,
        "notes": body.notes,
        "version": p.version,
    })
    p.review_history = history

    await db.commit()

    return {"success": True, "status": p.status}
