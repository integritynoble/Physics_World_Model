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

    return templates.TemplateResponse(request, "instruments.html", {
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


@router.get("/{instrument_id}", response_class=HTMLResponse)
async def instrument_detail_page(
    instrument_id: str,
    request: Request,
    user: Optional[User] = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    """Instrument detail page — shows InstrumentCard metadata and drift trend chart."""
    result = await db.execute(
        select(Instrument).where(Instrument.instrument_id == instrument_id)
    )
    instrument = result.scalar_one_or_none()
    if instrument is None:
        from fastapi import HTTPException as _HTTPException
        raise _HTTPException(status_code=404, detail=f"Instrument '{instrument_id}' not found")

    # Build drift trend data from card_data calibration history, or generate placeholder
    card = instrument.card_data or {}
    cal_history = card.get("calibration_history", [])

    # If no history stored, generate a simulated trend from calibration_data
    if not cal_history and card.get("calibration_data"):
        import datetime as _dt
        cal = card["calibration_data"]
        base_date = cal.get("calibration_date", "2025-01-01")
        try:
            d0 = _dt.date.fromisoformat(base_date)
        except Exception:
            d0 = _dt.date(2025, 1, 1)
        import random as _random
        _random.seed(hash(instrument_id))
        baseline = float(cal.get("cnr_baseline", 15.0))
        cal_history = []
        for i in range(12):
            d = d0 + _dt.timedelta(days=i * 30)
            drift = _random.gauss(0, baseline * 0.01)
            cal_history.append({
                "date": d.isoformat(),
                "cnr": round(baseline + drift * i * 0.1, 2),
                "noise_std": round(float(cal.get("noise_std_hu", 3.5)) + _random.gauss(0, 0.1), 2),
            })

    return templates.TemplateResponse(request, "instrument_detail.html", {
        "user": user,
        "instrument": instrument,
        "cal_history": cal_history,
        "card": card,
    })
