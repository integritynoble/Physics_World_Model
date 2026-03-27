"""Claim review workflow — human review of scaffolded ClaimCards.

Endpoints:
  GET  /claims/queue          — list pending ClaimCards
  GET  /claims/{claim_id}     — view one claim
  POST /claims/{claim_id}/approve  — approve to leaderboard (Draft tier)
  POST /claims/{claim_id}/reject   — reject with reason
  POST /claims/{claim_id}/reproduce — mark as independently reproduced
  POST /claims/{claim_id}/demote   — demote back to draft tier
  POST /claims/scaffold       — manually scaffold a new ClaimCard
"""

from __future__ import annotations

import json
import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from pwm_platform.auth.dependencies import get_current_user, get_optional_user, require_role
from pwm_platform.db.models import User

router = APIRouter(prefix="/claims", tags=["Claim Review"])

# Role requirements:
#   - View queue: any logged-in user
#   - Scaffold: any logged-in user
#   - Approve/Reject: admin or reviewer only
#   - Reproduce: admin or reviewer only
_reviewer_required = require_role("reviewer", "admin")

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
async def list_claims(status: str = "all", user: Optional[User] = Depends(get_optional_user)):
    """List all claims in the review queue. Visible to any logged-in user."""
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
async def scaffold_claim(
    request: Request,
    user: Optional[User] = Depends(get_optional_user),
):
    """Manually scaffold a new ClaimCard. Accepts both form data and JSON."""
    # Parse from form data or JSON
    content_type = request.headers.get("content-type", "")
    if "form" in content_type:
        form = await request.form()
        arxiv_id = form.get("arxiv_id", "")
        title = form.get("title", "")
        authors_str = form.get("authors", "")
        modality = form.get("modality", "")
        method = form.get("method", "")
        source_url = form.get("source_url", "")
        try:
            claimed_psnr = float(form.get("claimed_psnr")) if form.get("claimed_psnr") else None
        except (ValueError, TypeError):
            claimed_psnr = None
        try:
            claimed_ssim = float(form.get("claimed_ssim")) if form.get("claimed_ssim") else None
        except (ValueError, TypeError):
            claimed_ssim = None
    else:
        body = await request.json()
        arxiv_id = body.get("arxiv_id", "")
        title = body.get("title", "")
        authors_str = body.get("authors", "")
        modality = body.get("modality", "")
        method = body.get("method", "")
        source_url = body.get("source_url", "")
        claimed_psnr = body.get("claimed_psnr")
        claimed_ssim = body.get("claimed_ssim")

    # Extract explicit claim_type from form/JSON, or auto-detect from title prefix
    if "form" in content_type:
        explicit_type = form.get("claim_type", "")
    else:
        explicit_type = body.get("claim_type", "")

    if explicit_type:
        claim_type = explicit_type
    else:
        # Auto-detect claim_type from title prefix
        title_lower = title.lower()
        if title_lower.startswith("[red-team]"):
            claim_type = "red_team"
        elif title_lower.startswith("[reproduce]"):
            claim_type = "reproduction_issue"
        elif title_lower.startswith("[missing]"):
            claim_type = "maintainer_scaffold"
        elif title_lower.startswith("[test]") or title_lower.startswith("[demo]"):
            claim_type = "test"
        elif title_lower.startswith("[dispute]"):
            claim_type = "trust_dispute"
        else:
            claim_type = "paper_claim"

    claim_id = f"claim_manual_{arxiv_id.replace('.', '_').replace('/', '_') or datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    claim = {
        "claim_id": claim_id,
        "source_type": "arxiv" if arxiv_id else "manual",
        "source_id": arxiv_id,
        "source_url": source_url or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""),
        "title": title,
        "authors": [a.strip() for a in authors_str.split(",") if a.strip()],
        "modality": modality,
        "method": method,
        "claimed_psnr": claimed_psnr,
        "claimed_ssim": claimed_ssim,
        "claim_type": claim_type,
        "trust_tier": "draft",
        "status": "pending_review",
        "scaffolded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "history": [{"action": "scaffolded", "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()}],
    }
    _save_claim(claim)
    return claim


class BatchScaffoldRequest(BaseModel):
    arxiv_ids: str  # newline- or comma-separated arXiv IDs
    modality: str = ""
    default_method: str = ""


@router.post("/scaffold-batch")
async def scaffold_batch(req: BatchScaffoldRequest):
    """Scaffold multiple ClaimCards from a list of arXiv IDs."""
    import re as _re
    raw = _re.split(r"[\n,;]+", req.arxiv_ids)
    ids = [x.strip() for x in raw if x.strip()]
    if not ids:
        raise HTTPException(400, "No arXiv IDs provided")
    if len(ids) > 50:
        raise HTTPException(400, "Maximum 50 arXiv IDs per batch")

    created = []
    skipped = []
    for arxiv_id in ids:
        arxiv_id = arxiv_id.replace("arxiv:", "").replace("arXiv:", "").strip()
        claim_id = f"claim_manual_{arxiv_id.replace('.', '_').replace('/', '_')}"
        p = CLAIMS_DIR / f"{claim_id}.json"
        if p.exists():
            skipped.append(arxiv_id)
            continue
        claim = {
            "claim_id": claim_id,
            "source_type": "arxiv",
            "source_id": arxiv_id,
            "source_url": f"https://arxiv.org/abs/{arxiv_id}",
            "title": f"[Auto-scaffolded] arXiv:{arxiv_id}",
            "authors": [],
            "modality": req.modality,
            "method": req.default_method,
            "claimed_psnr": None,
            "claimed_ssim": None,
            "claim_type": "paper_claim",
            "trust_tier": "draft",
            "status": "pending_review",
            "scaffolded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "history": [{"action": "batch_scaffolded", "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()}],
        }
        _save_claim(claim)
        created.append(arxiv_id)

    return {"created": created, "skipped": skipped, "total_created": len(created)}


class AssignRequest(BaseModel):
    assignee: str = ""  # username or empty to unassign


@router.post("/{claim_id}/assign")
async def assign_claim(claim_id: str, body: AssignRequest):
    """Assign a claim to a reviewer/curator for follow-up."""
    claim = _load_claim(claim_id)
    claim["assigned_to"] = body.assignee
    claim["assigned_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    claim["history"] = claim.get("history", [])
    claim["history"].append({
        "action": "assigned",
        "to": body.assignee,
        "timestamp": claim["assigned_at"],
    })
    _save_claim(claim)
    return {"claim_id": claim_id, "assigned_to": body.assignee}


@router.post("/{claim_id}/approve")
async def approve_claim(claim_id: str, action: ReviewAction, user: User = Depends(get_current_user)):
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
async def reject_claim(claim_id: str, action: ReviewAction, user: User = Depends(get_current_user)):
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


@router.post("/{claim_id}/demote")
async def demote_claim(claim_id: str, action: ReviewAction, user: User = Depends(_reviewer_required)):
    """Demote an approved claim back to draft tier. Requires admin or reviewer role."""
    claim = _load_claim(claim_id)
    if claim.get("status") != "approved":
        raise HTTPException(409, "Claim must be approved before demotion")
    previous_tier = claim.get("trust_tier", "draft")
    if previous_tier == "draft":
        raise HTTPException(400, "Claim is already at Draft tier; nothing to demote.")
    claim["trust_tier"] = "draft"
    claim["demoted_by"] = action.reviewer
    claim["demoted_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    claim["history"] = claim.get("history", [])
    claim["history"].append({"action": "demoted", "by": action.reviewer, "reason": action.reason,
                              "from_tier": previous_tier, "timestamp": claim["demoted_at"]})
    _save_claim(claim)
    return claim


@router.post("/{claim_id}/reproduce")
async def mark_reproduced(claim_id: str, action: ReviewAction, user: User = Depends(_reviewer_required)):
    """Mark a claim as independently reproduced. Requires admin or reviewer role."""
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
