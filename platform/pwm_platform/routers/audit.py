"""
Audit Log — admin view of significant platform actions.

Endpoints:
  GET  /admin/audit      — audit log page (HTML, admin only)
  GET  /admin/audit/api  — audit log JSON (admin only)
  POST /admin/audit/api  — append an audit entry (internal use)
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
from pwm_platform.db.models import AuditLog, User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/audit", tags=["Audit"])

templates = Jinja2Templates(directory="pwm_platform/templates")


class AuditEntryRequest(BaseModel):
    action: str
    resource_type: str = ""
    resource_id: str = ""
    detail: str = ""


async def log_action(
    db: AsyncSession,
    action: str,
    actor_id: Optional[int] = None,
    actor_name: str = "",
    resource_type: str = "",
    resource_id: str = "",
    detail: str = "",
):
    """Helper to write an audit log entry. Import and call from other routers."""
    entry = AuditLog(
        action=action,
        actor_id=actor_id,
        actor_name=actor_name,
        resource_type=resource_type,
        resource_id=resource_id,
        detail=detail,
    )
    db.add(entry)
    # Don't commit here — caller is responsible for transaction


@router.get("", response_class=HTMLResponse)
async def audit_log_page(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    action: str = "",
    limit: int = 100,
):
    """Admin audit log page."""
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    stmt = select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit)
    if action:
        stmt = stmt.where(AuditLog.action == action)
    result = await db.execute(stmt)
    entries = result.scalars().all()

    # Distinct action types for filter
    action_result = await db.execute(select(AuditLog.action).distinct())
    action_types = sorted(r[0] for r in action_result.all() if r[0])

    return templates.TemplateResponse("audit_log.html", {
        "request": request,
        "user": user,
        "entries": entries,
        "action_types": action_types,
        "action_filter": action,
    })


@router.get("/api")
async def audit_log_api(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    action: str = "",
    limit: int = 100,
):
    """Audit log JSON API (admin only)."""
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    stmt = select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit)
    if action:
        stmt = stmt.where(AuditLog.action == action)
    result = await db.execute(stmt)
    entries = result.scalars().all()

    return {
        "entries": [
            {
                "id": e.id,
                "action": e.action,
                "actor_id": e.actor_id,
                "actor_name": e.actor_name,
                "resource_type": e.resource_type,
                "resource_id": e.resource_id,
                "detail": e.detail,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in entries
        ],
        "total": len(entries),
    }


@router.post("/api")
async def append_audit_entry(
    body: AuditEntryRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Append an audit log entry (admin or reviewer only)."""
    if user.role not in ("admin", "reviewer"):
        raise HTTPException(status_code=403, detail="Reviewer access required")

    await log_action(
        db,
        action=body.action,
        actor_id=user.id,
        actor_name=user.username,
        resource_type=body.resource_type,
        resource_id=body.resource_id,
        detail=body.detail,
    )
    await db.commit()
    return {"success": True}
