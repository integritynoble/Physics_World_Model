"""Trust tier promotion workflow.

Implements the trust tier state machine:
    Draft -> Author-confirmed -> Reproduced -> Certified

Endpoints
---------
POST /trust/promote/{submission_id}
    Request promotion to next tier. Requires appropriate role.
    Body: {"target_tier": "author_confirmed", "comment": "..."}

POST /trust/demote/{submission_id}
    Demote a submission back to Draft tier.
    Body: {"reason": "reproducibility_failure", "reviewer_id": "..."}

GET /trust/status/{submission_id}
    Get current trust tier + gate verdicts for a submission.

POST /trust/dispute/{submission_id}
    Flag a submission as reviewer-disputed.
    Body: {"reason": "..."}
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from pwm_platform.db.database import get_db
from pwm_platform.db.models import ChallengeSubmission

router = APIRouter(prefix="/trust", tags=["trust"])

_TIER_ORDER = ["rejected", "draft", "author_confirmed", "reproduced", "certified"]
_PROMOTION_RULES = {
    "author_confirmed": {
        "from": "draft",
        "description": "Author has reviewed and accepted the PWM result.",
    },
    "reproduced": {
        "from": "author_confirmed",
        "description": "An independent party has reproduced the result within declared tolerance.",
    },
    "certified": {
        "from": "reproduced",
        "description": "Full Judge (S1-S4 + domain gates) passed AND evidence package complete.",
    },
}


class PromoteRequest(BaseModel):
    target_tier: str
    comment: str = ""
    reviewer_id: str = ""


class DemoteRequest(BaseModel):
    reason: str = "reproducibility_failure"
    reviewer_id: str = ""


class DisputeRequest(BaseModel):
    reason: str
    reviewer_id: str = ""


class TrustStatusResponse(BaseModel):
    submission_id: str
    trust_tier: str
    gate_verdicts: Optional[dict]
    can_promote_to: Optional[str]
    promotion_rule: Optional[str]


@router.get("/status/{submission_id}", response_model=TrustStatusResponse)
async def get_trust_status(submission_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ChallengeSubmission).where(ChallengeSubmission.submission_id == submission_id)
    )
    sub = result.scalar_one_or_none()
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")

    current_tier = sub.trust_tier or "draft"
    next_tier = _next_tier(current_tier)
    rule = _PROMOTION_RULES.get(next_tier, {}).get("description") if next_tier else None

    return TrustStatusResponse(
        submission_id=submission_id,
        trust_tier=current_tier,
        gate_verdicts=sub.gate_verdicts,
        can_promote_to=next_tier,
        promotion_rule=rule,
    )


@router.post("/promote/{submission_id}")
async def promote_trust_tier(
    submission_id: str,
    body: PromoteRequest,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ChallengeSubmission).where(ChallengeSubmission.submission_id == submission_id)
    )
    sub = result.scalar_one_or_none()
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")

    current_tier = sub.trust_tier or "draft"
    target = body.target_tier

    # Validate promotion path
    rule = _PROMOTION_RULES.get(target)
    if rule is None:
        raise HTTPException(status_code=400, detail=f"Invalid target tier: {target!r}")
    if current_tier != rule["from"]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Cannot promote from '{current_tier}' to '{target}'. "
                f"Expected current tier: '{rule['from']}'."
            ),
        )

    # Block Certified if safety_brake risk flag is set
    if target == "certified":
        gate_verdicts = sub.gate_verdicts or {}
        if gate_verdicts.get("risk_flags") and "safety_brake" in gate_verdicts["risk_flags"]:
            raise HTTPException(
                status_code=400,
                detail="Cannot reach Certified tier: unresolved safety_brake risk flag.",
            )

    sub.trust_tier = target
    await db.commit()

    return {
        "submission_id": submission_id,
        "previous_tier": current_tier,
        "new_tier": target,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "comment": body.comment,
        "reviewer_id": body.reviewer_id,
    }


@router.post("/demote/{submission_id}")
async def demote_to_draft(
    submission_id: str,
    body: DemoteRequest,
    db: AsyncSession = Depends(get_db),
):
    """Demote a submission back to Draft tier.

    Allowed when: reproducibility fails or a gate that was previously
    waived is now enforced. Records the demotion reason in gate_verdicts.
    """
    result = await db.execute(
        select(ChallengeSubmission).where(ChallengeSubmission.submission_id == submission_id)
    )
    sub = result.scalar_one_or_none()
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")

    previous_tier = sub.trust_tier or "draft"
    if previous_tier == "draft":
        raise HTTPException(
            status_code=400,
            detail="Submission is already at Draft tier; nothing to demote.",
        )

    # Record demotion metadata in gate_verdicts
    verdicts = dict(sub.gate_verdicts or {})
    demotion_record = {
        "from_tier": previous_tier,
        "reason": body.reason,
        "reviewer_id": body.reviewer_id,
        "demoted_at": datetime.now(timezone.utc).isoformat(),
    }
    demotions = list(verdicts.get("demotions", []))
    demotions.append(demotion_record)
    verdicts["demotions"] = demotions
    sub.gate_verdicts = verdicts
    sub.trust_tier = "draft"
    await db.commit()

    return {
        "submission_id": submission_id,
        "previous_tier": previous_tier,
        "new_tier": "draft",
        "reason": body.reason,
        "demoted_at": demotion_record["demoted_at"],
        "reviewer_id": body.reviewer_id,
    }


@router.post("/dispute/{submission_id}")
async def flag_disputed(
    submission_id: str,
    body: DisputeRequest,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ChallengeSubmission).where(ChallengeSubmission.submission_id == submission_id)
    )
    sub = result.scalar_one_or_none()
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")

    # Add reviewer_disputed to gate_verdicts risk_flags
    verdicts = dict(sub.gate_verdicts or {})
    flags = list(verdicts.get("risk_flags", []))
    if "reviewer_disputed" not in flags:
        flags.append("reviewer_disputed")
    verdicts["risk_flags"] = flags
    verdicts["dispute_reason"] = body.reason
    verdicts["disputed_by"] = body.reviewer_id
    verdicts["disputed_at"] = datetime.now(timezone.utc).isoformat()
    sub.gate_verdicts = verdicts
    await db.commit()

    return {
        "submission_id": submission_id,
        "trust_tier": sub.trust_tier,
        "risk_flags": flags,
        "dispute_reason": body.reason,
    }


def _next_tier(current: str) -> Optional[str]:
    for target, rule in _PROMOTION_RULES.items():
        if rule["from"] == current:
            return target
    return None


@router.get("/contributor-economy-status")
async def contributor_economy_status(db: AsyncSession = Depends(get_db)):
    """Return current contributor economy activation status.

    Activation trigger: 10+ RunBundles at Draft+ from 3+ distinct contributors.
    """
    try:
        result = await db.execute(
            select(ChallengeSubmission).where(
                ChallengeSubmission.trust_tier.in_(
                    ["draft", "author_confirmed", "reproduced", "certified"]
                )
            )
        )
        submissions = result.scalars().all()
        bundle_count = len(submissions)
        distinct_contributors = len(
            set(s.submitted_by for s in submissions if s.submitted_by)
        )
    except Exception:
        bundle_count = 0
        distinct_contributors = 0

    activated = bundle_count >= 10 and distinct_contributors >= 3
    return {
        "activated": activated,
        "bundle_count": bundle_count,
        "distinct_contributors": distinct_contributors,
        "threshold_bundles": 10,
        "threshold_contributors": 3,
    }
