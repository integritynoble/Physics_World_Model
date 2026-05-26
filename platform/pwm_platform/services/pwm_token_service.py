"""
PWM Token Service — off-chain ledger for the PWM protocol reward token.

Contributors earn PWM tokens when their submissions (principle / spec /
benchmark / solution) are promoted from testnet (trust_tier in {draft,
author_confirmed, reproduced}) to mainnet (trust_tier == 'certified')
by a founder.

This service is the single source of truth for off-chain PWM balances.
Eventual settlement to the on-chain contract is handled by `agent-contracts`.
"""

from __future__ import annotations

import logging
import secrets
from typing import Optional

from fastapi import HTTPException
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from pwm_platform.db.models import PWMTokenAccount, PWMTokenTransaction

logger = logging.getLogger(__name__)


# ── Reward schedule ──────────────────────────────────────────────────────
#
# Reward amounts per artifact type when promoted testnet → mainnet.
# Tuned to favor the rarest, highest-leverage artifacts (principles).
#
PROMOTION_REWARDS: dict[str, float] = {
    "principle": 1000.0,
    "spec": 500.0,
    "benchmark": 300.0,
    "solution": 100.0,
    "reconstruction": 100.0,  # alias used by ChallengeSubmission.submission_type
    "algorithm": 100.0,       # alias
    "dataset": 200.0,         # alias for benchmark dataset
}

DEFAULT_PROMOTION_REWARD = 100.0


class PWMTokenService:
    """Stateless service — each method receives its own DB session."""

    # ── Account ──────────────────────────────────────────────────────────

    async def get_or_create_account(
        self, user_id: int, db: AsyncSession
    ) -> PWMTokenAccount:
        """Return the user's PWM token account, creating one if needed."""
        result = await db.execute(
            select(PWMTokenAccount).where(PWMTokenAccount.user_id == user_id)
        )
        account = result.scalar_one_or_none()
        if account is None:
            account = PWMTokenAccount(user_id=user_id, balance=0.0, lifetime_earned=0.0)
            db.add(account)
            await db.commit()
            await db.refresh(account)
        return account

    async def get_balance(self, user_id: int, db: AsyncSession) -> dict:
        """Return current balance + lifetime earned for a user."""
        account = await self.get_or_create_account(user_id, db)
        return {
            "user_id": user_id,
            "balance": account.balance,
            "lifetime_earned": account.lifetime_earned,
            "on_chain_address": account.on_chain_address,
        }

    async def list_transactions(
        self,
        user_id: int,
        db: AsyncSession,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """Return the user's transaction history, most-recent first."""
        result = await db.execute(
            select(PWMTokenTransaction)
            .where(PWMTokenTransaction.user_id == user_id)
            .order_by(desc(PWMTokenTransaction.created_at))
            .limit(limit)
            .offset(offset)
        )
        txns = result.scalars().all()
        return [_txn_to_dict(t) for t in txns]

    # ── Award / spend ────────────────────────────────────────────────────

    async def award_for_promotion(
        self,
        *,
        user_id: int,
        submission_id: str,
        artifact_type: str,
        awarded_by: Optional[int],
        db: AsyncSession,
    ) -> PWMTokenTransaction:
        """Award PWM tokens for a testnet → mainnet promotion.

        Idempotent per submission_id: returns the existing transaction if
        this submission has already been awarded.
        """
        # Idempotency check: don't award twice for the same submission
        result = await db.execute(
            select(PWMTokenTransaction).where(
                PWMTokenTransaction.submission_id == submission_id,
                PWMTokenTransaction.transaction_type == "award_promotion",
            )
        )
        existing = result.scalar_one_or_none()
        if existing is not None:
            logger.info(
                "PWM token award already exists for submission %s (txn %s)",
                submission_id,
                existing.transaction_id,
            )
            return existing

        amount = PROMOTION_REWARDS.get(artifact_type, DEFAULT_PROMOTION_REWARD)
        description = (
            f"Promotion reward: {artifact_type} (submission {submission_id}) "
            f"verified to mainnet"
        )
        return await self._credit(
            user_id=user_id,
            amount=amount,
            transaction_type="award_promotion",
            description=description,
            submission_id=submission_id,
            artifact_type=artifact_type,
            awarded_by=awarded_by,
            db=db,
        )

    async def award_manual(
        self,
        *,
        user_id: int,
        amount: float,
        description: str,
        awarded_by: int,
        db: AsyncSession,
    ) -> PWMTokenTransaction:
        """Manual award by a founder/admin (e.g., bug bounty, grant)."""
        if amount <= 0:
            raise HTTPException(status_code=400, detail="amount must be positive")
        return await self._credit(
            user_id=user_id,
            amount=amount,
            transaction_type="award_manual",
            description=description,
            submission_id=None,
            artifact_type=None,
            awarded_by=awarded_by,
            db=db,
        )

    async def adjust(
        self,
        *,
        user_id: int,
        amount: float,
        description: str,
        adjusted_by: int,
        db: AsyncSession,
    ) -> PWMTokenTransaction:
        """Admin adjustment (positive or negative). Used for clawbacks/refunds."""
        account = await self.get_or_create_account(user_id, db)
        new_balance = account.balance + amount
        if new_balance < 0:
            raise HTTPException(
                status_code=400,
                detail=f"Adjustment would result in negative balance ({new_balance})",
            )

        return await self._record(
            user_id=user_id,
            amount=amount,
            transaction_type="adjust",
            description=description,
            submission_id=None,
            artifact_type=None,
            awarded_by=adjusted_by,
            db=db,
        )

    async def set_on_chain_address(
        self, user_id: int, address: str, db: AsyncSession
    ) -> PWMTokenAccount:
        """Set the on-chain wallet address for the user (for future settlement)."""
        account = await self.get_or_create_account(user_id, db)
        account.on_chain_address = address
        await db.commit()
        await db.refresh(account)
        return account

    # ── Internal ─────────────────────────────────────────────────────────

    async def _credit(
        self,
        *,
        user_id: int,
        amount: float,
        transaction_type: str,
        description: str,
        submission_id: Optional[str],
        artifact_type: Optional[str],
        awarded_by: Optional[int],
        db: AsyncSession,
    ) -> PWMTokenTransaction:
        """Internal helper: credit tokens (positive amount only)."""
        if amount <= 0:
            raise ValueError("_credit requires a positive amount")
        return await self._record(
            user_id=user_id,
            amount=amount,
            transaction_type=transaction_type,
            description=description,
            submission_id=submission_id,
            artifact_type=artifact_type,
            awarded_by=awarded_by,
            db=db,
            is_credit=True,
        )

    async def _record(
        self,
        *,
        user_id: int,
        amount: float,
        transaction_type: str,
        description: str,
        submission_id: Optional[str],
        artifact_type: Optional[str],
        awarded_by: Optional[int],
        db: AsyncSession,
        is_credit: bool = False,
    ) -> PWMTokenTransaction:
        """Atomic record + balance update."""
        account = await self.get_or_create_account(user_id, db)

        account.balance += amount
        if is_credit:
            account.lifetime_earned += amount

        txn = PWMTokenTransaction(
            transaction_id=secrets.token_urlsafe(16),
            user_id=user_id,
            amount=amount,
            transaction_type=transaction_type,
            description=description,
            submission_id=submission_id,
            artifact_type=artifact_type,
            awarded_by=awarded_by,
            balance_after=account.balance,
        )
        db.add(txn)
        await db.commit()
        await db.refresh(txn)

        logger.info(
            "PWM token %s: user=%s amount=%+.2f balance=%.2f (%s)",
            transaction_type,
            user_id,
            amount,
            account.balance,
            description,
        )
        return txn


def _txn_to_dict(t: PWMTokenTransaction) -> dict:
    return {
        "transaction_id": t.transaction_id,
        "amount": t.amount,
        "transaction_type": t.transaction_type,
        "description": t.description,
        "submission_id": t.submission_id,
        "artifact_type": t.artifact_type,
        "balance_after": t.balance_after,
        "created_at": t.created_at.isoformat() if t.created_at else None,
    }


# Module-level singleton
pwm_token_service = PWMTokenService()
