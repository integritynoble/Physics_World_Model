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

import base64
import logging
import os
import secrets
from typing import Optional

from cryptography.fernet import Fernet

from fastapi import HTTPException
from sqlalchemy import desc, func, select
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

# ── Auto-award (on automatic ACCEPT) ─────────────────────────────────────
# Small fixed amount credited automatically the moment a submission passes the
# S1–S4 acceptance gate. Deliberately tiny because acceptance is automated and
# spammable; the founder's 0.1–5 manual award (award_for_modification) tops up
# genuinely good work on review. A per-user DAILY CAP bounds total auto-minting.
AUTO_ACCEPT_REWARD = 0.1
AUTO_ACCEPT_DAILY_CAP = 2.0   # max PWM a single user can auto-earn per UTC day

DEFAULT_PROMOTION_REWARD = 100.0

# ── Service costs ────────────────────────────────────────────────────────
#
# PWM tokens required to access premium platform features.
# Shallow review is always free; deep review requires PAPER_REVIEW_COSTS["deep"].
#
PAPER_REVIEW_COSTS: dict[str, float] = {
    "deep": 10.0,     # full multi-reviewer simulation
    "shallow": 0.0,   # single reviewer, always free
}

# Provider wallet that receives paper review payments.
# 5th wallet — owned by the developer of the paper review feature.
PAPER_REVIEW_PROVIDER_WALLET = "0xa53F7e7Bc6B0Cc182d048217646082DDB2DacfE3"

# Modification submission reward range (founder sets amount at review time)
MODIFICATION_REWARD_RANGE = (0.1, 5.0)

# ── Provider / mining-pool split (record-only; settled on-chain by M6) ────
# Every spend is attributed: provider_split of the amount to the solution /
# compute / review PROVIDER wallet, and the remainder to the mining POOL.
# These shares are RECORDED on the spend transaction (PWMTokenTransaction
# .provider_amount / .pool_amount) for the settlement relayer to credit on
# Base — the off-chain ledger debits only the user (it is user-keyed and has
# no pool account, so it cannot hold provider/pool balances). Override the
# provider share via PWM_PROVIDER_SPLIT_BPS (basis points; default 9000 = 90%,
# leaving 1000 = 10% for the pool).
DEFAULT_PROVIDER_SPLIT_BPS = 9000


# ── Wallet encryption ────────────────────────────────────────────────────

def _fernet() -> Fernet:
    """Derive a stable Fernet key from SECRET_KEY (env var).

    Fernet requires a 32-byte URL-safe base64 key.  We hash the server
    secret with SHA-256 and base64-encode the result so the derivation is
    deterministic and reversible only with the same SECRET_KEY.
    """
    import hashlib
    raw = os.environ.get("SECRET_KEY", "dev-secret-change-me")
    digest = hashlib.sha256(raw.encode()).digest()
    key = base64.urlsafe_b64encode(digest)
    return Fernet(key)


def generate_wallet() -> tuple[str, str]:
    """Generate a new Ethereum wallet.

    Returns (address, encrypted_private_key_hex).
    The private key is Fernet-encrypted with the server's SECRET_KEY so it
    is safe to store in the database.  Decrypt with decrypt_wallet_key().
    """
    from eth_account import Account
    acct = Account.create()
    private_key_hex = acct.key.hex()
    encrypted = _fernet().encrypt(private_key_hex.encode()).decode()
    return acct.address, encrypted


def decrypt_wallet_key(encrypted: str) -> str:
    """Decrypt a private key stored by generate_wallet(). Returns hex string."""
    return _fernet().decrypt(encrypted.encode()).decode()


def provider_split_bps() -> int:
    """Provider share of a spend, in basis points (0–10000). Env-overridable."""
    raw = os.environ.get("PWM_PROVIDER_SPLIT_BPS")
    if raw is None:
        return DEFAULT_PROVIDER_SPLIT_BPS
    try:
        bps = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_PROVIDER_SPLIT_BPS
    return bps if 0 <= bps <= 10000 else DEFAULT_PROVIDER_SPLIT_BPS


def split_spend(amount: float, bps: int) -> tuple[float, float]:
    """Split a spent amount into (provider_amount, pool_amount).

    provider_amount = bps/10000 of `amount`; pool_amount is the remainder, so
    the two always sum to `amount` (the split neither creates nor loses PWM).
    """
    provider_amount = round(amount * bps / 10000.0, 6)
    pool_amount = round(amount - provider_amount, 6)
    return provider_amount, pool_amount


class PWMTokenService:
    """Stateless service — each method receives its own DB session."""

    # ── Account ──────────────────────────────────────────────────────────

    async def provision_pwm_account(
        self,
        user_id: int,
        db: AsyncSession,
        *,
        existing_wallet: Optional[str] = None,
    ) -> PWMTokenAccount:
        """Create (or return) the PWM account for a newly registered user.

        Called once at signup for every auth path.  If the user already
        signed in via SIWE and has a wallet address, pass it as
        `existing_wallet` — no custodial key is generated.  Otherwise a
        fresh Ethereum wallet is generated and the encrypted private key is
        stored in `wallet_private_key_enc`.

        Idempotent: safe to call more than once (returns existing account).
        """
        result = await db.execute(
            select(PWMTokenAccount).where(PWMTokenAccount.user_id == user_id)
        )
        account = result.scalar_one_or_none()
        if account is not None:
            return account

        if existing_wallet:
            address = existing_wallet
            enc_key = None
        else:
            address, enc_key = generate_wallet()

        account = PWMTokenAccount(
            user_id=user_id,
            balance=0.0,
            lifetime_earned=0.0,
            on_chain_address=address,
            wallet_private_key_enc=enc_key,
        )
        db.add(account)
        await db.commit()
        await db.refresh(account)
        logger.info(
            "Provisioned PWM account for user %s → %s (%s)",
            user_id,
            address,
            "external" if existing_wallet else "custodial",
        )
        return account

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
            "is_custodial": account.wallet_private_key_enc is not None,
        }

    async def export_private_key(self, user_id: int, db: AsyncSession) -> str:
        """Decrypt and return the custodial private key for this user.

        Raises 404 if no account exists, 400 if the wallet is external (no key stored).
        """
        result = await db.execute(
            select(PWMTokenAccount).where(PWMTokenAccount.user_id == user_id)
        )
        account = result.scalar_one_or_none()
        if account is None:
            raise HTTPException(status_code=404, detail="PWM account not found")
        if not account.wallet_private_key_enc:
            raise HTTPException(
                status_code=400,
                detail="This wallet was provided externally — no private key is stored by the platform.",
            )
        return decrypt_wallet_key(account.wallet_private_key_enc)

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

    async def award_auto_accept(
        self,
        *,
        user_id: int,
        submission_id: str,
        artifact_type: str,
        db: AsyncSession,
    ) -> Optional[PWMTokenTransaction]:
        """Auto-credit a small fixed reward when a submission is ACCEPTED.

        Anti-spam, in order:
          1. Idempotent per submission_id — never pay twice for the same one.
          2. Per-user daily cap — sum of today's auto-awards may not exceed
             AUTO_ACCEPT_DAILY_CAP; returns None (no credit) once hit.
        The founder's manual award_for_modification (0.1–5) tops up good work.
        Returns the transaction, or None if skipped (already awarded / capped).
        """
        from datetime import datetime, timezone

        # 1. idempotency
        existing = (
            await db.execute(
                select(PWMTokenTransaction).where(
                    PWMTokenTransaction.submission_id == submission_id,
                    PWMTokenTransaction.transaction_type == "award_auto_accept",
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            return existing

        # 2. per-user daily cap (UTC day)
        start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        today_total = (
            await db.execute(
                select(func.coalesce(func.sum(PWMTokenTransaction.amount), 0.0)).where(
                    PWMTokenTransaction.user_id == user_id,
                    PWMTokenTransaction.transaction_type == "award_auto_accept",
                    PWMTokenTransaction.created_at >= start,
                )
            )
        ).scalar() or 0.0
        if today_total + AUTO_ACCEPT_REWARD > AUTO_ACCEPT_DAILY_CAP:
            logger.info(
                "auto-accept award skipped for %s: daily cap reached (user %s, %.2f today)",
                submission_id, user_id, today_total,
            )
            return None

        return await self._credit(
            user_id=user_id,
            amount=AUTO_ACCEPT_REWARD,
            transaction_type="award_auto_accept",
            description=f"Auto-award: {artifact_type} {submission_id} passed S1–S4 acceptance",
            submission_id=submission_id,
            artifact_type=artifact_type,
            awarded_by=None,
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

    async def award_for_modification(
        self,
        *,
        user_id: int,
        submission_id: str,
        amount: float,
        comment: str,
        awarded_by: int,
        db: AsyncSession,
    ) -> PWMTokenTransaction:
        """Award PWM tokens for a reviewed modification submission (0.1–5.0 PWM).

        Idempotent per submission_id — re-awarding the same modification returns
        the existing transaction without crediting again.

        Also stamps `pwm_awarded` on the PWMSubmission row if found.
        """
        if not (0.1 <= amount <= 5.0):
            raise HTTPException(
                status_code=400,
                detail=f"Modification reward must be between 0.1 and 5.0 PWM (got {amount})",
            )

        # Idempotency check
        result = await db.execute(
            select(PWMTokenTransaction).where(
                PWMTokenTransaction.submission_id == submission_id,
                PWMTokenTransaction.transaction_type == "award_modification",
            )
        )
        existing = result.scalar_one_or_none()
        if existing is not None:
            logger.info(
                "Modification award already exists for submission %s (txn %s)",
                submission_id,
                existing.transaction_id,
            )
            return existing

        # Stamp pwm_awarded on PWMSubmission if the model exists in this codebase
        try:
            from pwm_platform.db.models import PWMSubmission
            sub_result = await db.execute(
                select(PWMSubmission).where(PWMSubmission.submission_id == submission_id)
            )
            sub = sub_result.scalar_one_or_none()
            if sub is not None:
                sub.pwm_awarded = amount
                if sub.status in ("testnet", "reviewing"):
                    sub.status = "mainnet"
        except ImportError:
            pass  # PWMSubmission not available in this codebase variant

        description = f"Modification reward: {submission_id}"
        if comment:
            description += f" — {comment}"

        return await self._credit(
            user_id=user_id,
            amount=amount,
            transaction_type="award_modification",
            description=description,
            submission_id=submission_id,
            artifact_type="modification",
            awarded_by=awarded_by,
            db=db,
        )

    async def spend(
        self,
        *,
        user_id: int,
        amount: float,
        purpose: str,
        provider_wallet: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        db: AsyncSession,
    ) -> PWMTokenTransaction:
        """Deduct PWM tokens from a user's balance for a platform service.

        Args:
            user_id: The user spending tokens.
            amount: Positive amount to deduct.
            purpose: Human-readable label (e.g. 'paper_review_deep').
            provider_wallet: Wallet address that receives the payment (for audit trail).
            idempotency_key: Optional unique key — if a spend transaction with
                this key already exists for this user, return it without deducting again.
            db: Async DB session.

        Raises:
            HTTPException 402 if the user's balance is insufficient.
        """
        if amount <= 0:
            raise HTTPException(status_code=400, detail="amount must be positive")

        # Idempotency: don't charge twice for the same operation
        if idempotency_key:
            result = await db.execute(
                select(PWMTokenTransaction).where(
                    PWMTokenTransaction.user_id == user_id,
                    PWMTokenTransaction.submission_id == idempotency_key,
                    PWMTokenTransaction.transaction_type == "spend",
                )
            )
            existing = result.scalar_one_or_none()
            if existing is not None:
                logger.info(
                    "PWM spend already recorded for key %s (txn %s)",
                    idempotency_key,
                    existing.transaction_id,
                )
                return existing

        # Balance check
        account = await self.get_or_create_account(user_id, db)
        if account.balance < amount:
            raise HTTPException(
                status_code=402,
                detail=(
                    f"Insufficient PWM balance. Required: {amount}, "
                    f"available: {account.balance:.2f}"
                ),
            )

        wallet_note = f" → provider {provider_wallet}" if provider_wallet else ""
        description = f"{purpose}{wallet_note}"

        # 90/10 split — recorded on the spend row for on-chain settlement (M6).
        # The user is debited the full `amount`; provider_amount + pool_amount
        # == amount and describe how that spend is to be credited on Base.
        split_bps = provider_split_bps()
        provider_amount, pool_amount = split_spend(amount, split_bps)

        return await self._record(
            user_id=user_id,
            amount=-amount,
            transaction_type="spend",
            description=description,
            submission_id=idempotency_key,
            artifact_type=purpose,
            awarded_by=None,
            db=db,
            provider_wallet=provider_wallet,
            provider_amount=provider_amount,
            pool_amount=pool_amount,
            provider_split_bps=split_bps,
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
        provider_wallet: Optional[str] = None,
        provider_amount: Optional[float] = None,
        pool_amount: Optional[float] = None,
        provider_split_bps: Optional[int] = None,
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
            provider_wallet=provider_wallet,
            provider_amount=provider_amount,
            pool_amount=pool_amount,
            provider_split_bps=provider_split_bps,
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
        "provider_wallet": t.provider_wallet,
        "provider_amount": t.provider_amount,
        "pool_amount": t.pool_amount,
        "provider_split_bps": t.provider_split_bps,
        "created_at": t.created_at.isoformat() if t.created_at else None,
    }


# Module-level singleton
pwm_token_service = PWMTokenService()
