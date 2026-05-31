"""
PWM Token Router — endpoints for the protocol reward token.

Endpoints:
    GET    /api/v1/pwm-token/balance            — current user's PWM balance
    GET    /api/v1/pwm-token/transactions       — current user's ledger
    POST   /api/v1/pwm-token/wallet             — set on-chain wallet address
    POST   /api/v1/pwm-token/award              — founder/admin manual award
    POST   /api/v1/pwm-token/adjust             — founder/admin balance adjustment
    POST   /api/v1/pwm-token/promote-to-mainnet — promote a submission and award tokens
    GET    /api/v1/pwm-token/leaderboard        — top contributors by lifetime earned (public)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from pwm_platform.auth.dependencies import get_current_user, require_role
from pwm_platform.db.database import get_db
from pwm_platform.db.models import ChallengeSubmission, PWMTokenAccount, User
from pwm_platform.services.pwm_token_service import (
    PROMOTION_REWARDS,
    DEFAULT_PROMOTION_REWARD,
    PAPER_REVIEW_COSTS,
    PAPER_REVIEW_PROVIDER_WALLET,
    MODIFICATION_REWARD_RANGE,
    pwm_token_service,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/pwm-token", tags=["PWM Token"])


# ── Request / response schemas ───────────────────────────────────────────


class WalletRequest(BaseModel):
    address: str = Field(..., min_length=1, max_length=255)


class AwardRequest(BaseModel):
    user_id: int
    amount: float = Field(..., gt=0)
    description: str = Field(..., min_length=1, max_length=500)


class AdjustRequest(BaseModel):
    user_id: int
    amount: float  # can be positive or negative
    description: str = Field(..., min_length=1, max_length=500)


class PromoteToMainnetRequest(BaseModel):
    submission_id: str
    comment: str = ""


class AwardModificationRequest(BaseModel):
    submission_id: str = Field(..., min_length=1, max_length=128)
    amount: float = Field(
        ...,
        ge=0.1,
        le=5.0,
        description="PWM tokens to award (0.1–5.0). Founder sets this at review time.",
    )
    comment: str = Field("", max_length=500)


class SpendRequest(BaseModel):
    amount: float = Field(..., gt=0, description="PWM tokens to spend")
    purpose: str = Field(..., min_length=1, max_length=100)
    provider_wallet: Optional[str] = Field(
        None,
        max_length=255,
        description="Wallet address receiving the payment (audit trail)",
    )
    idempotency_key: Optional[str] = Field(
        None,
        max_length=128,
        description="Unique key to prevent double-charging the same operation",
    )


# ── User-facing endpoints (own data) ─────────────────────────────────────


@router.get("/balance")
async def get_balance(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the authenticated user's PWM token balance."""
    data = await pwm_token_service.get_balance(user.id, db)
    return {"success": True, **data}


@router.get("/transactions")
async def get_transactions(
    limit: int = Query(50, le=200, ge=1),
    offset: int = Query(0, ge=0),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the authenticated user's PWM token ledger."""
    txns = await pwm_token_service.list_transactions(
        user.id, db, limit=limit, offset=offset
    )
    return {"success": True, "transactions": txns}


@router.post("/wallet")
async def set_wallet(
    body: WalletRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Set the on-chain wallet address for the authenticated user."""
    account = await pwm_token_service.set_on_chain_address(user.id, body.address, db)
    return {
        "success": True,
        "on_chain_address": account.on_chain_address,
    }


# ── Spending endpoints ───────────────────────────────────────────────────


@router.post("/spend")
async def spend_tokens(
    body: SpendRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Deduct PWM tokens from the authenticated user's balance for a platform service.

    Returns HTTP 402 if the balance is insufficient.
    Idempotent when `idempotency_key` is provided — safe to retry.
    """
    txn = await pwm_token_service.spend(
        user_id=user.id,
        amount=body.amount,
        purpose=body.purpose,
        provider_wallet=body.provider_wallet,
        idempotency_key=body.idempotency_key,
        db=db,
    )
    balance_before = txn.balance_after + body.amount
    return {
        "success": True,
        "transaction_id": txn.transaction_id,
        "amount_spent": body.amount,
        "balance_before": balance_before,
        "balance_after": txn.balance_after,
        "purpose": body.purpose,
        "provider_wallet": body.provider_wallet,
    }


@router.get("/paper-access")
async def paper_access(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the user's paper review access level based on their PWM balance.

    Does NOT deduct tokens — call /spend to charge before running deep review.
    """
    data = await pwm_token_service.get_balance(user.id, db)
    balance = data["balance"]
    deep_cost = PAPER_REVIEW_COSTS["deep"]
    can_afford_deep = balance >= deep_cost
    return {
        "success": True,
        "balance": balance,
        "deep_cost": deep_cost,
        "shallow_cost": PAPER_REVIEW_COSTS["shallow"],
        "can_afford_deep": can_afford_deep,
        "review_level": "deep" if can_afford_deep else "shallow",
        "provider_wallet": PAPER_REVIEW_PROVIDER_WALLET,
        "message": (
            None
            if can_afford_deep
            else f"You need {deep_cost} PWM for deep review. "
            f"Your balance: {balance:.2f} PWM. "
            "Earn PWM by submitting verified principles, specs, benchmarks, or solutions."
        ),
    }


# ── Founder / admin endpoints ────────────────────────────────────────────


@router.post("/award")
async def manual_award(
    body: AwardRequest,
    admin: User = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db),
):
    """Manually award PWM tokens to a user (founder/admin only)."""
    txn = await pwm_token_service.award_manual(
        user_id=body.user_id,
        amount=body.amount,
        description=body.description,
        awarded_by=admin.id,
        db=db,
    )
    return {
        "success": True,
        "transaction_id": txn.transaction_id,
        "balance_after": txn.balance_after,
    }


@router.post("/award-modification")
async def award_modification(
    body: AwardModificationRequest,
    admin: User = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db),
):
    """Award PWM tokens for a reviewed modification submission (0.1–5.0 PWM).

    The founder sets the exact amount based on the quality of the modification.
    Idempotent: re-submitting the same submission_id returns the existing award.
    Also promotes the PWMSubmission row to status='mainnet' if not already there.
    """
    # Resolve submitter from PWMSubmission, fallback to ChallengeSubmission
    user_id: Optional[int] = None

    try:
        from pwm_platform.db.models import PWMSubmission
        result = await db.execute(
            select(PWMSubmission).where(PWMSubmission.submission_id == body.submission_id)
        )
        sub = result.scalar_one_or_none()
        if sub is not None:
            user_id = sub.submitted_by_id
    except ImportError:
        pass  # PWMSubmission not available in this codebase variant

    if user_id is None:
        result2 = await db.execute(
            select(ChallengeSubmission).where(
                ChallengeSubmission.submission_id == body.submission_id
            )
        )
        legacy = result2.scalar_one_or_none()
        if legacy is not None:
            user_id = legacy.submitted_by

    if user_id is None:  # not found in either model
        raise HTTPException(
            status_code=404,
            detail=f"Submission '{body.submission_id}' not found or has no owner",
        )

    txn = await pwm_token_service.award_for_modification(
        user_id=user_id,
        submission_id=body.submission_id,
        amount=body.amount,
        comment=body.comment,
        awarded_by=admin.id,
        db=db,
    )
    return {
        "success": True,
        "submission_id": body.submission_id,
        "user_id": user_id,
        "transaction_id": txn.transaction_id,
        "amount_awarded": txn.amount,
        "balance_after": txn.balance_after,
        "reward_range": list(MODIFICATION_REWARD_RANGE),
    }


@router.post("/adjust")
async def manual_adjust(
    body: AdjustRequest,
    admin: User = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db),
):
    """Admin balance adjustment (positive or negative). Use for clawbacks/refunds."""
    txn = await pwm_token_service.adjust(
        user_id=body.user_id,
        amount=body.amount,
        description=body.description,
        adjusted_by=admin.id,
        db=db,
    )
    return {
        "success": True,
        "transaction_id": txn.transaction_id,
        "balance_after": txn.balance_after,
    }


@router.post("/promote-to-mainnet")
async def promote_to_mainnet(
    body: PromoteToMainnetRequest,
    admin: User = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db),
):
    """Promote a submission from testnet to mainnet (trust_tier=certified)
    and atomically award PWM tokens to the submitter.

    Idempotent: if the submission is already certified, no duplicate reward is given.
    """
    result = await db.execute(
        select(ChallengeSubmission).where(
            ChallengeSubmission.submission_id == body.submission_id
        )
    )
    sub = result.scalar_one_or_none()
    if sub is None:
        raise HTTPException(status_code=404, detail="Submission not found")
    if sub.submitted_by is None:
        raise HTTPException(status_code=400, detail="Submission has no owner — cannot reward")

    previous_tier = sub.trust_tier or "draft"
    was_already_certified = previous_tier == "certified"

    # Promote to certified (mainnet)
    sub.trust_tier = "certified"
    sub.status = "approved"
    sub.reviewer_id = admin.id
    sub.reviewed_at = datetime.now(timezone.utc)
    if body.comment:
        sub.review_notes = body.comment

    # Award tokens (idempotent — won't double-award)
    artifact_type = sub.submission_type or "solution"
    txn = await pwm_token_service.award_for_promotion(
        user_id=sub.submitted_by,
        submission_id=body.submission_id,
        artifact_type=artifact_type,
        awarded_by=admin.id,
        db=db,
    )

    return {
        "success": True,
        "submission_id": body.submission_id,
        "previous_tier": previous_tier,
        "new_tier": "certified",
        "deploy_status": "mainnet",
        "was_already_certified": was_already_certified,
        "reward_transaction_id": txn.transaction_id,
        "reward_amount": txn.amount,
        "submitter_balance_after": txn.balance_after,
    }


# ── Public endpoints ─────────────────────────────────────────────────────


@router.get("/leaderboard")
async def leaderboard(
    limit: int = Query(20, le=100, ge=1),
    db: AsyncSession = Depends(get_db),
):
    """Top contributors ranked by lifetime PWM tokens earned. Public."""
    result = await db.execute(
        select(PWMTokenAccount, User)
        .join(User, User.id == PWMTokenAccount.user_id)
        .where(PWMTokenAccount.lifetime_earned > 0)
        .order_by(desc(PWMTokenAccount.lifetime_earned))
        .limit(limit)
    )
    rows = result.all()
    leaderboard = [
        {
            "rank": idx + 1,
            "username": user.username,
            "lifetime_earned": account.lifetime_earned,
            "balance": account.balance,
        }
        for idx, (account, user) in enumerate(rows)
    ]
    return {"success": True, "leaderboard": leaderboard}


# ── Reward schedule (public) ─────────────────────────────────────────────


@router.get("/reward-schedule")
async def reward_schedule():
    """Return the PWM reward amounts for each artifact type. Public."""
    return {
        "success": True,
        "rewards": PROMOTION_REWARDS,
        "default": DEFAULT_PROMOTION_REWARD,
        "currency": "PWM",
        "trigger": "submission promoted to trust_tier='certified' (mainnet)",
    }
