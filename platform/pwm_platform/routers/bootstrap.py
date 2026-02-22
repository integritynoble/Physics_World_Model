"""
Bootstrap Router — new modality bootstrap proposals.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from pwm_platform.auth.dependencies import get_current_user, require_role
from pwm_platform.db.database import get_db
from pwm_platform.db.models import BootstrapProposal, User

router = APIRouter(prefix="/api/v1/bootstrap", tags=["Bootstrap"])


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


@router.post("")
async def create_proposal(
    body: BootstrapCreateRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Submit a new modality bootstrap proposal."""
    proposal_id = f"bp-{uuid.uuid4().hex[:12]}"

    proposal = BootstrapProposal(
        proposal_id=proposal_id,
        modality_key=body.modality_key,
        display_name=body.display_name or body.modality_key,
        submitted_by=user.id,
        status="draft",
        physics_class=body.physics_class,
        forward_model_family=body.forward_model_family,
        sensor_type=body.sensor_type,
        source_type=body.source_type,
        geometry=body.geometry,
        noise_model=body.noise_model,
    )
    db.add(proposal)
    await db.commit()
    await db.refresh(proposal)

    # TODO: run similarity engine + generate templates

    return {
        "success": True,
        "proposal_id": proposal_id,
        "status": "draft",
    }


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
