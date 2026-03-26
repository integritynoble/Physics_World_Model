"""
Instrument Registry — InstrumentCard upload, browse, and calibration tracking.

Endpoints:
  GET  /instruments           — instrument registry page (HTML)
  GET  /instruments/api       — list instruments (JSON)
  POST /instruments/api/register — register a new InstrumentCard (JSON)
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from pwm_platform.auth.dependencies import get_current_user, get_optional_user
from pwm_platform.db.database import get_db
from pwm_platform.db.models import Instrument, User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/instruments", tags=["Instruments"])

templates = Jinja2Templates(directory="pwm_platform/templates")


class InstrumentRegisterRequest(BaseModel):
    instrument_id: str
    name: str
    modality: str
    lab: str = ""
    institution: str = ""
    manufacturer: str = ""
    model_number: str = ""
    serial_number: str = ""
    description: str = ""
    drift_budget_pct: float = 0.0
    contact_email: str = ""
    is_public: bool = True
    card_data: dict = {}


@router.get("", response_class=HTMLResponse)
async def instruments_page(
    request: Request,
    user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
    modality: str = "",
):
    """Instrument Registry browser page."""
    stmt = select(Instrument).order_by(Instrument.created_at.desc()).limit(200)
    if modality:
        stmt = stmt.where(Instrument.modality == modality)

    result = await db.execute(stmt)
    instruments = result.scalars().all()

    # Distinct modalities for filter dropdown
    mod_result = await db.execute(select(Instrument.modality).distinct())
    modalities = sorted(r[0] for r in mod_result.all() if r[0])

    return templates.TemplateResponse("instruments.html", {
        "request": request,
        "user": user,
        "instruments": instruments,
        "modalities": modalities,
        "modality_filter": modality,
    })


@router.get("/api")
async def list_instruments(
    modality: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """List registered instruments (JSON)."""
    stmt = select(Instrument).order_by(Instrument.created_at.desc()).limit(200)
    if modality:
        stmt = stmt.where(Instrument.modality == modality)
    result = await db.execute(stmt)
    instruments = result.scalars().all()
    return {
        "instruments": [
            {
                "instrument_id": i.instrument_id,
                "name": i.name,
                "modality": i.modality,
                "lab": i.lab,
                "institution": i.institution,
                "manufacturer": i.manufacturer,
                "model_number": i.model_number,
                "serial_number": i.serial_number,
                "description": i.description,
                "drift_budget_pct": i.drift_budget_pct,
                "contact_email": i.contact_email,
                "is_public": i.is_public,
                "created_at": i.created_at.isoformat() if i.created_at else None,
            }
            for i in instruments
        ],
        "total": len(instruments),
    }


@router.post("/api/register")
async def register_instrument(
    body: InstrumentRegisterRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Register a new InstrumentCard."""
    existing = await db.execute(
        select(Instrument).where(Instrument.instrument_id == body.instrument_id)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=409,
            detail=f"Instrument '{body.instrument_id}' already registered",
        )

    inst = Instrument(
        instrument_id=body.instrument_id,
        name=body.name,
        modality=body.modality,
        lab=body.lab,
        institution=body.institution,
        manufacturer=body.manufacturer,
        model_number=body.model_number,
        serial_number=body.serial_number,
        description=body.description,
        drift_budget_pct=body.drift_budget_pct,
        contact_email=body.contact_email,
        is_public=body.is_public,
        card_data=body.card_data,
        uploaded_by=user.id,
    )
    db.add(inst)
    await db.commit()
    await db.refresh(inst)

    logger.info("Instrument registered: %s by user %s", body.instrument_id, user.id)
    return {"instrument_id": inst.instrument_id, "created": True}
