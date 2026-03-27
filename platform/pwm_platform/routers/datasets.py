"""
Datasets Router — browse and inspect registered datasets.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from pwm_platform.auth.dependencies import get_current_user
from pwm_platform.db.database import get_db
from pwm_platform.db.models import Dataset, User

router = APIRouter(prefix="/api/v1/datasets", tags=["Datasets"])


@router.get("")
async def list_datasets(
    modality: str | None = None,
    data_type: str | None = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = 100,
    offset: int = 0,
):
    """List registered datasets, optionally filtered by modality or type."""
    stmt = select(Dataset).order_by(Dataset.created_at.desc())
    if modality:
        stmt = stmt.where(Dataset.modality == modality)
    if data_type:
        stmt = stmt.where(Dataset.data_type == data_type)
    stmt = stmt.limit(limit).offset(offset)

    result = await db.execute(stmt)
    datasets = result.scalars().all()

    return {
        "datasets": [
            {
                "dataset_id": d.dataset_id,
                "modality": d.modality,
                "data_type": d.data_type,
                "description": d.description,
                "num_samples": d.num_samples,
                "x_shape": d.x_shape,
                "y_shape": d.y_shape,
                "version": d.version,
                "is_public": d.is_public,
                "tags": d.tags or [],
            }
            for d in datasets
        ],
        "total": len(datasets),
    }


from pydantic import BaseModel as _BaseModel


class DatasetRegisterRequest(_BaseModel):
    dataset_id: str
    version: str = "1.0.0"
    modality: str
    data_type: str  # simulation / real / benchmark
    description: str = ""
    source: str = ""
    license: str = "internal"
    num_samples: int = 0
    gcs_prefix: str = ""
    is_public: bool = False
    tags: list = []


@router.post("/register")
async def register_dataset(
    body: DatasetRegisterRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Register a new DatasetCard (Dataset Steward or admin)."""
    from sqlalchemy import select as _select
    existing = await db.execute(_select(Dataset).where(Dataset.dataset_id == body.dataset_id))
    if existing.scalar_one_or_none():
        from fastapi import HTTPException as _HTTPException
        raise _HTTPException(status_code=409, detail=f"Dataset '{body.dataset_id}' already registered")

    ds = Dataset(
        dataset_id=body.dataset_id,
        version=body.version,
        modality=body.modality,
        data_type=body.data_type,
        description=body.description,
        source=body.source,
        license=body.license,
        num_samples=body.num_samples,
        gcs_prefix=body.gcs_prefix,
        is_public=body.is_public,
        tags=body.tags,
        created_by=user.id,
    )
    db.add(ds)
    await db.commit()
    await db.refresh(ds)
    return {"dataset_id": ds.dataset_id, "created": True}


@router.get("/history/{dataset_id}")
async def dataset_version_history(
    dataset_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get version history for a dataset (all versions sharing the same base ID)."""
    # Strip version suffix (e.g. "cassi_sim_v2" → "cassi_sim")
    import re as _re
    base_id = _re.sub(r'_v\d+(\.\d+)*$', '', dataset_id)

    stmt = (
        select(Dataset)
        .where(Dataset.dataset_id.like(f"{base_id}%"))
        .order_by(Dataset.created_at.asc())
    )
    result = await db.execute(stmt)
    versions = result.scalars().all()

    return {
        "base_id": base_id,
        "versions": [
            {
                "dataset_id": d.dataset_id,
                "version": d.version,
                "modality": d.modality,
                "num_samples": d.num_samples,
                "is_public": d.is_public,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            }
            for d in versions
        ],
        "total": len(versions),
    }


@router.get("/{dataset_id}")
async def get_dataset(
    dataset_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get full dataset metadata."""
    result = await db.execute(
        select(Dataset).where(Dataset.dataset_id == dataset_id)
    )
    ds = result.scalar_one_or_none()
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return {
        "dataset_id": ds.dataset_id,
        "version": ds.version,
        "modality": ds.modality,
        "data_type": ds.data_type,
        "description": ds.description,
        "source": ds.source,
        "license": ds.license,
        "num_samples": ds.num_samples,
        "x_shape": ds.x_shape,
        "y_shape": ds.y_shape,
        "gcs_prefix": ds.gcs_prefix,
        "experiment_spec": ds.experiment_spec,
        "calibration_ref": ds.calibration_ref,
        "manifest_ref": ds.manifest_ref,
        "tags": ds.tags or [],
        "is_public": ds.is_public,
        "created_at": ds.created_at.isoformat() if ds.created_at else None,
    }
