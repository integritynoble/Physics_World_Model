"""Claim review workflow — human review of scaffolded ClaimCards.

Endpoints:
  GET  /claims/queue          — list pending ClaimCards
  GET  /claims/{claim_id}     — view one claim
  POST /claims/{claim_id}/approve  — approve to leaderboard (Draft tier)
  POST /claims/{claim_id}/reject   — reject with reason
  POST /claims/{claim_id}/reproduce — mark as independently reproduced
  POST /claims/scaffold       — manually scaffold a new ClaimCard
"""

from __future__ import annotations

import json
import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

router = APIRouter(prefix="/claims", tags=["Claim Review"])

# Store claims in a simple JSON directory (in production: database)
CLAIMS_DIR = Path("/tmp/pwm_claim_queue")
CLAIMS_DIR.mkdir(parents=True, exist_ok=True)


class ScaffoldRequest(BaseModel):
    arxiv_id: str = ""
    title: str = ""
    authors: str = ""  # comma-separated
    modality: str = ""
    method: str = ""
    claimed_psnr: Optional[float] = None
    claimed_ssim: Optional[float] = None
    source_url: str = ""


class ReviewAction(BaseModel):
    reviewer: str = "admin"
    reason: str = ""


def _load_claim(claim_id: str) -> dict:
    p = CLAIMS_DIR / f"{claim_id}.json"
    if not p.exists():
        raise HTTPException(404, f"Claim {claim_id} not found")
    return json.loads(p.read_text())


def _save_claim(claim: dict):
    p = CLAIMS_DIR / f"{claim['claim_id']}.json"
    p.write_text(json.dumps(claim, indent=2, default=str))


@router.get("/queue")
async def list_claims(status: str = "all"):
    """List all claims in the review queue."""
    claims = []
    for f in sorted(CLAIMS_DIR.glob("*.json")):
        try:
            c = json.loads(f.read_text())
            if status == "all" or c.get("status") == status:
                claims.append(c)
        except Exception:
            continue
    return {"claims": claims, "total": len(claims)}


@router.get("/{claim_id}")
async def get_claim(claim_id: str):
    """View one claim."""
    return _load_claim(claim_id)


@router.post("/scaffold")
async def scaffold_claim(req: ScaffoldRequest):
    """Manually scaffold a new ClaimCard."""
    claim_id = f"claim_manual_{req.arxiv_id.replace('.', '_').replace('/', '_') or datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    claim = {
        "claim_id": claim_id,
        "source_type": "arxiv" if req.arxiv_id else "manual",
        "source_id": req.arxiv_id,
        "source_url": req.source_url or (f"https://arxiv.org/abs/{req.arxiv_id}" if req.arxiv_id else ""),
        "title": req.title,
        "authors": [a.strip() for a in req.authors.split(",") if a.strip()],
        "modality": req.modality,
        "method": req.method,
        "claimed_psnr": req.claimed_psnr,
        "claimed_ssim": req.claimed_ssim,
        "trust_tier": "draft",
        "status": "pending_review",
        "scaffolded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "history": [{"action": "scaffolded", "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()}],
    }
    _save_claim(claim)
    return claim


@router.post("/{claim_id}/approve")
async def approve_claim(claim_id: str, action: ReviewAction):
    """Approve a claim to appear on the leaderboard at Draft tier."""
    claim = _load_claim(claim_id)
    if claim.get("status") == "approved":
        raise HTTPException(409, "Already approved")
    claim["status"] = "approved"
    claim["trust_tier"] = "draft"
    claim["reviewed_by"] = action.reviewer
    claim["reviewed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    claim["history"] = claim.get("history", [])
    claim["history"].append({"action": "approved", "by": action.reviewer, "reason": action.reason,
                              "timestamp": claim["reviewed_at"]})
    _save_claim(claim)
    return claim


@router.post("/{claim_id}/reject")
async def reject_claim(claim_id: str, action: ReviewAction):
    """Reject a claim with reason."""
    claim = _load_claim(claim_id)
    claim["status"] = "rejected"
    claim["trust_tier"] = "rejected"
    claim["reviewed_by"] = action.reviewer
    claim["reviewed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    claim["rejection_reason"] = action.reason
    claim["history"] = claim.get("history", [])
    claim["history"].append({"action": "rejected", "by": action.reviewer, "reason": action.reason,
                              "timestamp": claim["reviewed_at"]})
    _save_claim(claim)
    return claim


@router.post("/{claim_id}/reproduce")
async def mark_reproduced(claim_id: str, action: ReviewAction):
    """Mark a claim as independently reproduced."""
    claim = _load_claim(claim_id)
    if claim.get("status") != "approved":
        raise HTTPException(409, "Claim must be approved before reproduction")
    claim["trust_tier"] = "reproduced"
    claim["reproduced_by"] = action.reviewer
    claim["reproduced_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    claim["history"] = claim.get("history", [])
    claim["history"].append({"action": "reproduced", "by": action.reviewer, "reason": action.reason,
                              "timestamp": claim["reproduced_at"]})
    _save_claim(claim)
    return claim
