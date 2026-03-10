"""
PWM Billing Service — credit management, subscription lifecycle, usage enforcement.

Adapted from CompareGPT's payment model with PWM-specific concepts:
- run_credits: monthly allowance for reference runs
- report_credits: monthly allowance for PWM validation reports
- Subscriptions via Stripe (recurring) and WeChat Pay (one-time credit packs)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from pwm_platform.db.models import (
    CreditAccount,
    CreditTransaction,
    PaymentOrder,
    Subscription,
    User,
)

logger = logging.getLogger(__name__)


# ── Plan definitions ──────────────────────────────────────────────────────

PLANS: dict[str, dict] = {
    # ── Free ───────────────────────────────────────────────────────────
    "free": {
        "display_name": "Community",
        "price_monthly": 0,
        "price_yearly": 0,
        "run_credits_monthly": 3,
        "report_credits_monthly": 0,
        "max_dataset_size": "small",
        "features": [
            "3 reference runs / month",
            "Small datasets only",
            "Standard methods",
            "Basic output preview",
            "Public queue",
        ],
        "watermark": "For research preview only",
    },
    # ── Researcher ─────────────────────────────────────────────────────
    "researcher": {
        "display_name": "Researcher",
        "price_monthly": 49,
        "price_yearly": 490,
        "run_credits_monthly": 20,
        "report_credits_monthly": 5,
        "max_dataset_size": "medium",
        "features": [
            "20 reference runs / month",
            "Small–medium datasets",
            "Leaderboard-backed methods",
            "Downloadable reconstructions",
            "5 PWM validation reports / month",
            "Run history & reproducibility logs",
            "Priority queue",
        ],
        "watermark": None,
    },
    # ── Pro Lab ─────────────────────────────────────────────────────────
    "pro": {
        "display_name": "Pro Lab",
        "price_monthly": 199,
        "price_yearly": 1990,
        "run_credits_monthly": 100,
        "report_credits_monthly": 25,
        "max_dataset_size": "large",
        "features": [
            "100 reference runs / month",
            "Large dataset support",
            "Batch jobs",
            "25 PWM validation reports / month",
            "Method comparison mode",
            "Benchmark comparison tables",
            "Calibration / failure-diagnosis summary",
            "API access",
            "Exportable PDF / CSV / JSON reports",
            "Faster queue / priority support",
        ],
        "watermark": None,
    },
    # ── Team ───────────────────────────────────────────────────────────
    "team": {
        "display_name": "Team",
        "price_monthly": 799,
        "price_yearly": 7990,
        "run_credits_monthly": 500,
        "report_credits_monthly": 100,
        "max_dataset_size": "large",
        "max_seats": 10,
        "features": [
            "500 reference runs / month",
            "5–10 seats",
            "Shared workspace & project folders",
            "100 PWM validation reports / month",
            "Private datasets",
            "Role-based access",
            "API + integration support",
            "Higher compute priority",
            "Benchmark submission tools",
            "White-label report option",
        ],
        "watermark": None,
    },
    # ── Enterprise (custom pricing) ────────────────────────────────────
    "enterprise": {
        "display_name": "Enterprise",
        "price_monthly": 0,  # custom
        "price_yearly": 0,   # custom ($15k-$100k+)
        "run_credits_monthly": 9999,
        "report_credits_monthly": 9999,
        "max_dataset_size": "unlimited",
        "features": [
            "Private cloud / VPC / on-prem deployment",
            "Custom SLAs & modality support",
            "Internal benchmarking & private leaderboard",
            "Unlimited reports (contract-defined)",
            "SSO / audit trail / security review",
            "Dedicated support",
            "Custom evaluation protocol",
            "Procurement-friendly invoicing",
        ],
        "watermark": None,
    },
}

# ── Overage pricing (USD) ─────────────────────────────────────────────

OVERAGE_RUN_PRICES = {
    "small": 3.0,
    "medium": 10.0,
    "large": 25.0,
}

OVERAGE_REPORT_PRICES = {
    "standard": 25.0,
    "advanced": 75.0,
    "enterprise": 250.0,
}

# ── WeChat Credit Packs (one-time purchase bundles) ───────────────────
# Since WeChat Pay only supports one-time payments, we offer prepaid
# "Credit Packs" (充值包) that mirror subscription tiers.  Users buy packs
# and credits are valid for a set period.  This is the standard Chinese
# market model (similar to 视频VIP月卡, 季卡, 年卡).

WECHAT_CREDIT_PACKS: dict[str, dict] = {
    # ── Researcher packs ───────────────────────────────────────────────
    "researcher_month": {
        "display_name": "Researcher Monthly Pack",
        "display_name_zh": "研究员月度充值包",
        "plan_tier": "researcher",
        "price_usd": 49,
        "price_cny": 349,
        "run_credits": 20,
        "report_credits": 5,
        "validity_days": 30,
    },
    "researcher_quarter": {
        "display_name": "Researcher Quarterly Pack",
        "display_name_zh": "研究员季度充值包",
        "plan_tier": "researcher",
        "price_usd": 129,
        "price_cny": 929,
        "run_credits": 60,
        "report_credits": 15,
        "validity_days": 90,
    },
    "researcher_year": {
        "display_name": "Researcher Annual Pack",
        "display_name_zh": "研究员年度充值包",
        "plan_tier": "researcher",
        "price_usd": 490,
        "price_cny": 3490,
        "run_credits": 240,
        "report_credits": 60,
        "validity_days": 365,
    },
    # ── Pro packs ──────────────────────────────────────────────────────
    "pro_month": {
        "display_name": "Pro Lab Monthly Pack",
        "display_name_zh": "专业实验室月度充值包",
        "plan_tier": "pro",
        "price_usd": 199,
        "price_cny": 1399,
        "run_credits": 100,
        "report_credits": 25,
        "validity_days": 30,
    },
    "pro_quarter": {
        "display_name": "Pro Lab Quarterly Pack",
        "display_name_zh": "专业实验室季度充值包",
        "plan_tier": "pro",
        "price_usd": 529,
        "price_cny": 3799,
        "run_credits": 300,
        "report_credits": 75,
        "validity_days": 90,
    },
    "pro_year": {
        "display_name": "Pro Lab Annual Pack",
        "display_name_zh": "专业实验室年度充值包",
        "plan_tier": "pro",
        "price_usd": 1990,
        "price_cny": 13990,
        "run_credits": 1200,
        "report_credits": 300,
        "validity_days": 365,
    },
    # ── Team packs ─────────────────────────────────────────────────────
    "team_month": {
        "display_name": "Team Monthly Pack",
        "display_name_zh": "团队月度充值包",
        "plan_tier": "team",
        "price_usd": 799,
        "price_cny": 5699,
        "run_credits": 500,
        "report_credits": 100,
        "validity_days": 30,
    },
    "team_year": {
        "display_name": "Team Annual Pack",
        "display_name_zh": "团队年度充值包",
        "plan_tier": "team",
        "price_usd": 7990,
        "price_cny": 56990,
        "run_credits": 6000,
        "report_credits": 1200,
        "validity_days": 365,
    },
}

# ── Add-on pricing ────────────────────────────────────────────────────

ADDON_PRICES = {
    "private_data_retention": 20.0,     # $/month
    "fast_turnaround_queue": 50.0,      # $/month
    "extra_seat": 25.0,                 # $/month
    "premium_support": 200.0,           # $/month
}


# ═════════════════════════════════════════════════════════════════════════
#  Service class
# ═════════════════════════════════════════════════════════════════════════


class BillingService:
    """Core billing operations for PWM."""

    def __init__(self, db: AsyncSession):
        self.db = db

    # ── Credit account helpers ────────────────────────────────────────

    async def get_or_create_credit_account(self, user_id: int) -> CreditAccount:
        """Return (or lazily create) the user's credit account."""
        stmt = select(CreditAccount).where(CreditAccount.user_id == user_id)
        result = await self.db.execute(stmt)
        account = result.scalar_one_or_none()

        if account is None:
            account = CreditAccount(
                user_id=user_id,
                plan_tier="free",
                run_credits=PLANS["free"]["run_credits_monthly"],
                report_credits=PLANS["free"]["report_credits_monthly"],
                payment_status="free",
            )
            self.db.add(account)
            await self.db.commit()
            await self.db.refresh(account)
            logger.info("Created credit account for user_id=%s", user_id)

        return account

    async def get_account_balance(self, user_id: int) -> dict:
        """Public-facing balance summary."""
        acct = await self.get_or_create_credit_account(user_id)
        plan = PLANS.get(acct.plan_tier, PLANS["free"])
        return {
            "plan_tier": acct.plan_tier,
            "plan_display_name": plan["display_name"],
            "run_credits": acct.run_credits,
            "report_credits": acct.report_credits,
            "overage_run_credits": acct.overage_run_credits,
            "overage_report_credits": acct.overage_report_credits,
            "payment_status": acct.payment_status,
            "subscription_id": acct.subscription_id,
            "credits_expire_at": (
                acct.credits_expire_at.isoformat() if acct.credits_expire_at else None
            ),
            "legacy_credit_balance": acct.legacy_credit_balance,
        }

    # ── Credit consumption ────────────────────────────────────────────

    async def consume_run_credit(
        self, user_id: int, description: str = "Reference run"
    ) -> tuple[bool, str]:
        """
        Deduct 1 run credit.  Returns (success, message).
        Priority: plan credits → overage credits → deny.
        """
        acct = await self.get_or_create_credit_account(user_id)

        # Check expiry for credit-pack users
        if acct.credits_expire_at and datetime.now(timezone.utc) > acct.credits_expire_at:
            acct.run_credits = 0
            acct.report_credits = 0

        if acct.run_credits > 0:
            acct.run_credits -= 1
        elif acct.overage_run_credits > 0:
            acct.overage_run_credits -= 1
        else:
            return False, "No run credits remaining. Upgrade your plan or purchase a credit pack."

        # Log transaction
        txn = CreditTransaction(
            transaction_id=str(uuid4()),
            user_id=user_id,
            transaction_type="consume",
            credit_kind="run",
            amount=-1,
            description=description,
            remaining_run_credits=acct.run_credits,
            remaining_report_credits=acct.report_credits,
        )
        self.db.add(txn)
        await self.db.commit()
        return True, "OK"

    async def consume_report_credit(
        self, user_id: int, report_type: str = "standard", description: str = "Validation report"
    ) -> tuple[bool, str]:
        """Deduct 1 report credit."""
        acct = await self.get_or_create_credit_account(user_id)

        if acct.credits_expire_at and datetime.now(timezone.utc) > acct.credits_expire_at:
            acct.run_credits = 0
            acct.report_credits = 0

        if acct.report_credits > 0:
            acct.report_credits -= 1
        elif acct.overage_report_credits > 0:
            acct.overage_report_credits -= 1
        else:
            return False, "No report credits remaining. Upgrade your plan or purchase a credit pack."

        txn = CreditTransaction(
            transaction_id=str(uuid4()),
            user_id=user_id,
            transaction_type="consume",
            credit_kind="report",
            amount=-1,
            description=f"{description} ({report_type})",
            remaining_run_credits=acct.run_credits,
            remaining_report_credits=acct.report_credits,
        )
        self.db.add(txn)
        await self.db.commit()
        return True, "OK"

    # ── Legacy credit consumption (hidden tier, existing logic) ───────

    async def consume_legacy_credits(
        self, user_id: int, amount: float, description: str
    ) -> tuple[bool, str]:
        """Deduct from legacy credit_balance (used for hidden tier submissions)."""
        acct = await self.get_or_create_credit_account(user_id)
        if acct.legacy_credit_balance < amount:
            return False, "Insufficient credits"
        acct.legacy_credit_balance -= amount
        txn = CreditTransaction(
            transaction_id=str(uuid4()),
            user_id=user_id,
            transaction_type="consume",
            credit_kind="legacy",
            amount=-amount,
            description=description,
            remaining_run_credits=acct.run_credits,
            remaining_report_credits=acct.report_credits,
        )
        self.db.add(txn)
        await self.db.commit()
        return True, "OK"

    # ── Credit provisioning (after payment or subscription renewal) ───

    async def provision_plan_credits(
        self,
        user_id: int,
        plan_tier: str,
        subscription_id: Optional[str] = None,
        validity_days: Optional[int] = None,
        run_credits: Optional[int] = None,
        report_credits: Optional[int] = None,
    ) -> CreditAccount:
        """
        Provision credits for a user after payment.
        For Stripe subscriptions: called monthly on renewal.
        For WeChat credit packs: called once per purchase.
        """
        plan = PLANS.get(plan_tier, PLANS["free"])
        acct = await self.get_or_create_credit_account(user_id)

        acct.plan_tier = plan_tier
        acct.payment_status = "active" if plan_tier != "free" else "free"
        if subscription_id:
            acct.subscription_id = subscription_id

        # Set credits (use overrides if provided, e.g. for credit packs)
        new_runs = run_credits if run_credits is not None else plan["run_credits_monthly"]
        new_reports = report_credits if report_credits is not None else plan["report_credits_monthly"]

        acct.run_credits = new_runs
        acct.report_credits = new_reports

        # Set expiry for credit packs
        if validity_days:
            acct.credits_expire_at = datetime.now(timezone.utc) + timedelta(days=validity_days)
        else:
            # Stripe subscriptions: no expiry (renewed monthly)
            acct.credits_expire_at = None

        # Log transaction
        txn = CreditTransaction(
            transaction_id=str(uuid4()),
            user_id=user_id,
            transaction_type="provision",
            credit_kind="plan",
            amount=new_runs + new_reports,
            description=f"Plan provisioned: {plan['display_name']} ({plan_tier})",
            remaining_run_credits=acct.run_credits,
            remaining_report_credits=acct.report_credits,
        )
        self.db.add(txn)
        await self.db.commit()
        await self.db.refresh(acct)
        logger.info(
            "Provisioned %s credits for user_id=%s (plan=%s, runs=%d, reports=%d)",
            plan_tier, user_id, plan["display_name"], new_runs, new_reports,
        )
        return acct

    # ── Overage purchases ─────────────────────────────────────────────

    async def add_overage_credits(
        self,
        user_id: int,
        credit_kind: str,  # "run" or "report"
        amount: int,
        order_id: str,
        description: str = "Overage purchase",
    ) -> CreditAccount:
        """Add overage credits after an add-on purchase."""
        acct = await self.get_or_create_credit_account(user_id)
        if credit_kind == "run":
            acct.overage_run_credits += amount
        elif credit_kind == "report":
            acct.overage_report_credits += amount
        txn = CreditTransaction(
            transaction_id=str(uuid4()),
            user_id=user_id,
            transaction_type="purchase",
            credit_kind=credit_kind,
            amount=amount,
            description=description,
            remaining_run_credits=acct.run_credits + acct.overage_run_credits,
            remaining_report_credits=acct.report_credits + acct.overage_report_credits,
        )
        self.db.add(txn)
        await self.db.commit()
        return acct

    # ── Subscription cancellation ─────────────────────────────────────

    async def cancel_subscription(self, user_id: int) -> dict:
        """
        Cancel a Stripe subscription.  Credits remain until end of period.
        """
        acct = await self.get_or_create_credit_account(user_id)
        if not acct.subscription_id:
            return {"error": "No active subscription"}

        # Stripe cancellation is handled in the router (needs stripe import)
        # Here we just reset the account state
        acct.payment_status = "cancelled"
        # Don't zero credits yet — they run until period end
        await self.db.commit()
        return {"status": "cancelled", "subscription_id": acct.subscription_id}

    async def downgrade_to_free(self, user_id: int) -> CreditAccount:
        """Reset user to free tier (e.g. after subscription end)."""
        acct = await self.get_or_create_credit_account(user_id)
        acct.plan_tier = "free"
        acct.payment_status = "free"
        acct.subscription_id = None
        acct.run_credits = PLANS["free"]["run_credits_monthly"]
        acct.report_credits = PLANS["free"]["report_credits_monthly"]
        acct.credits_expire_at = None
        await self.db.commit()
        return acct

    # ── Order management ──────────────────────────────────────────────

    async def create_order(
        self,
        user_id: int,
        order_type: str,
        plan_tier: str,
        amount_usd: float,
        payment_method: str,
        pack_key: Optional[str] = None,
    ) -> PaymentOrder:
        """Create a payment order before redirecting to Stripe/WeChat."""
        order = PaymentOrder(
            order_id=str(uuid4()).replace("-", ""),
            user_id=user_id,
            order_type=order_type,
            plan_tier=plan_tier,
            pack_key=pack_key,
            amount_usd=amount_usd,
            payment_method=payment_method,
            status="pending",
        )
        self.db.add(order)
        await self.db.commit()
        await self.db.refresh(order)
        return order

    async def complete_order(self, order_id: str, payment_ref: str = "") -> Optional[PaymentOrder]:
        """Mark order as completed and provision credits."""
        stmt = select(PaymentOrder).where(PaymentOrder.order_id == order_id)
        result = await self.db.execute(stmt)
        order = result.scalar_one_or_none()
        if not order:
            logger.warning("Order not found: %s", order_id)
            return None
        if order.status == "completed":
            logger.warning("Order already completed: %s", order_id)
            return order

        order.status = "completed"
        order.payment_ref = payment_ref
        order.completed_at = datetime.now(timezone.utc)

        # Provision credits based on order type
        if order.order_type == "subscription":
            await self.provision_plan_credits(
                user_id=order.user_id,
                plan_tier=order.plan_tier,
                subscription_id=payment_ref,
            )
        elif order.order_type == "credit_pack":
            pack = WECHAT_CREDIT_PACKS.get(order.pack_key, {})
            if pack:
                await self.provision_plan_credits(
                    user_id=order.user_id,
                    plan_tier=pack["plan_tier"],
                    validity_days=pack["validity_days"],
                    run_credits=pack["run_credits"],
                    report_credits=pack["report_credits"],
                )
        elif order.order_type == "overage_run":
            await self.add_overage_credits(
                user_id=order.user_id,
                credit_kind="run",
                amount=order.credit_amount or 1,
                order_id=order.order_id,
            )
        elif order.order_type == "overage_report":
            await self.add_overage_credits(
                user_id=order.user_id,
                credit_kind="report",
                amount=order.credit_amount or 1,
                order_id=order.order_id,
            )

        await self.db.commit()
        logger.info("Order completed: %s (type=%s, user=%s)", order_id, order.order_type, order.user_id)
        return order

    # ── Transaction history ───────────────────────────────────────────

    async def get_transaction_history(
        self, user_id: int, limit: int = 50, offset: int = 0
    ) -> list[dict]:
        """Return recent credit transactions for a user."""
        stmt = (
            select(CreditTransaction)
            .where(CreditTransaction.user_id == user_id)
            .order_by(CreditTransaction.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        rows = result.scalars().all()
        return [
            {
                "id": t.transaction_id,
                "type": t.transaction_type,
                "kind": t.credit_kind,
                "amount": t.amount,
                "description": t.description,
                "remaining_runs": t.remaining_run_credits,
                "remaining_reports": t.remaining_report_credits,
                "created_at": t.created_at.isoformat() if t.created_at else None,
            }
            for t in rows
        ]

    async def get_payment_history(
        self, user_id: int, limit: int = 50
    ) -> list[dict]:
        """Return recent payment orders for a user."""
        stmt = (
            select(PaymentOrder)
            .where(PaymentOrder.user_id == user_id)
            .order_by(PaymentOrder.created_at.desc())
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        rows = result.scalars().all()
        return [
            {
                "order_id": o.order_id,
                "type": o.order_type,
                "plan": o.plan_tier,
                "amount_usd": o.amount_usd,
                "payment_method": o.payment_method,
                "status": o.status,
                "created_at": o.created_at.isoformat() if o.created_at else None,
                "completed_at": o.completed_at.isoformat() if o.completed_at else None,
            }
            for o in rows
        ]

    # ── Usage enforcement helpers ─────────────────────────────────────

    async def can_submit_run(self, user_id: int) -> tuple[bool, str]:
        """Check if user has available run credits."""
        acct = await self.get_or_create_credit_account(user_id)
        if acct.credits_expire_at and datetime.now(timezone.utc) > acct.credits_expire_at:
            return False, "Credits expired. Please renew your plan or purchase a credit pack."
        total = acct.run_credits + acct.overage_run_credits
        if total <= 0:
            return False, "No run credits remaining."
        return True, "OK"

    async def can_generate_report(self, user_id: int) -> tuple[bool, str]:
        """Check if user has available report credits."""
        acct = await self.get_or_create_credit_account(user_id)
        if acct.credits_expire_at and datetime.now(timezone.utc) > acct.credits_expire_at:
            return False, "Credits expired."
        total = acct.report_credits + acct.overage_report_credits
        if total <= 0:
            return False, "No report credits remaining."
        return True, "OK"

    def get_plan_info(self, plan_tier: str) -> dict:
        """Return plan details (no DB access)."""
        return PLANS.get(plan_tier, PLANS["free"])

    def get_all_plans(self) -> dict[str, dict]:
        """Return all plans."""
        return PLANS

    def get_wechat_credit_packs(self) -> dict[str, dict]:
        """Return all WeChat credit packs."""
        return WECHAT_CREDIT_PACKS
