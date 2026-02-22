"""
Modalities Router — list and inspect supported imaging modalities.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from pwm_platform.auth.dependencies import get_current_user
from pwm_platform.db.database import get_db
from pwm_platform.db.models import ModalityBasics, User

router = APIRouter(prefix="/api/v1/modalities", tags=["Modalities"])


@router.get("")
async def list_modalities(
    category: str | None = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all known modalities from the knowledge base."""
    stmt = select(ModalityBasics).order_by(ModalityBasics.display_name)
    if category:
        stmt = stmt.where(ModalityBasics.category == category)

    result = await db.execute(stmt)
    modalities = result.scalars().all()

    return {
        "modalities": [
            {
                "modality_key": m.modality_key,
                "display_name": m.display_name,
                "category": m.category,
                "physics_class": m.physics_class,
                "default_solver": m.default_solver,
                "tags": m.tags or [],
            }
            for m in modalities
        ],
        "total": len(modalities),
    }


@router.get("/{modality_key}")
async def get_modality(
    modality_key: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get detailed modality information."""
    result = await db.execute(
        select(ModalityBasics).where(ModalityBasics.modality_key == modality_key)
    )
    m = result.scalar_one_or_none()
    if m is None:
        raise HTTPException(status_code=404, detail="Modality not found")

    return {
        "modality_key": m.modality_key,
        "display_name": m.display_name,
        "category": m.category,
        "physics_class": m.physics_class,
        "forward_model_family": m.forward_model_family,
        "primitive_gates": m.primitive_gates,
        "wave_model": m.wave_model,
        "sensor_type": m.sensor_type,
        "source_type": m.source_type,
        "geometry": m.geometry,
        "typical_x_dims": m.typical_x_dims,
        "typical_y_dims": m.typical_y_dims,
        "calibration_params": m.calibration_params,
        "mismatch_modes": m.mismatch_modes,
        "noise_model": m.noise_model,
        "default_solver": m.default_solver,
        "evaluation_metrics": m.evaluation_metrics,
        "default_experiment_spec": m.default_experiment_spec,
        "default_operator_graph": m.default_operator_graph,
        "canonical_references": m.canonical_references,
        "tags": m.tags or [],
    }
