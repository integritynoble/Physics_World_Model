"""Knowledge base CRUD for ModalityRecord (backed by PostgreSQL via SQLAlchemy)."""
from __future__ import annotations
from typing import Optional


async def get_all_modalities(db) -> list:
    """Return all ModalityRecord rows."""
    from sqlalchemy import select
    from api.models.modality import ModalityRecord
    result = await db.execute(select(ModalityRecord))
    return result.scalars().all()


async def get_modality(db, modality_id: str):
    from api.models.modality import ModalityRecord
    return await db.get(ModalityRecord, modality_id)


async def upsert_modality(db, record_dict: dict) -> None:
    from api.models.modality import ModalityRecord
    existing = await db.get(ModalityRecord, record_dict["id"])
    if existing:
        for k, v in record_dict.items():
            setattr(existing, k, v)
    else:
        db.add(ModalityRecord(**record_dict))
    await db.commit()
