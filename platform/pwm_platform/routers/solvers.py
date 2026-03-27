"""
Solvers Router — register and list community SolverCards.

Built-in solvers come from the algorithm catalog (read-only).
Community solvers are stored as JSON files in /tmp/pwm_solvers/.
"""

from __future__ import annotations

import json
import datetime
import re
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from pwm_platform.auth.dependencies import get_optional_user
from pwm_platform.db.models import User

router = APIRouter(prefix="/api/v1/solvers", tags=["Solvers"])

SOLVERS_DIR = Path("/tmp/pwm_solvers")
SOLVERS_DIR.mkdir(parents=True, exist_ok=True)


class SolverRegisterRequest(BaseModel):
    name: str
    solver_type: str = "Deep Learning"
    modality: str
    description: str = ""
    code_url: str = ""
    paper_url: str = ""
    params: str = ""
    mask_aware: bool = False


@router.get("/community")
async def list_community_solvers():
    """List all community-registered solvers."""
    solvers = []
    for f in sorted(SOLVERS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            solvers.append(json.loads(f.read_text()))
        except Exception:
            continue
    return {"solvers": solvers, "total": len(solvers)}


@router.post("/register")
async def register_solver(
    body: SolverRegisterRequest,
    user: Optional[User] = Depends(get_optional_user),
):
    """Register a new community SolverCard."""
    if not body.name.strip():
        raise HTTPException(status_code=400, detail="name is required")
    if not body.modality.strip():
        raise HTTPException(status_code=400, detail="modality is required")

    slug = re.sub(r"[^a-z0-9]+", "_", body.name.lower()).strip("_")
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S")
    solver_id = f"solver_{slug}_{ts}"

    card = {
        "card_type": "SolverCard",
        "solver_id": solver_id,
        "name": body.name.strip(),
        "solver_type": body.solver_type,
        "modality": body.modality.lower().strip(),
        "description": body.description,
        "code_url": body.code_url,
        "paper_url": body.paper_url,
        "params": body.params,
        "mask_aware": body.mask_aware,
        "source": "community",
        "registered_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "registered_by": user.username if user else "anonymous",
    }

    p = SOLVERS_DIR / f"{solver_id}.json"
    p.write_text(json.dumps(card, indent=2, default=str))

    return card
