"""
PWM Subscription Router — subscription management, Stripe + WeChat Pay integration.

Endpoints:
  GET  /api/v1/subscription/balance          — current credit balance
  GET  /api/v1/subscription/transactions     — credit transaction history
  GET  /api/v1/subscription/payments         — payment order history
  POST /api/v1/subscription/checkout/stripe  — create Stripe checkout session
  POST /api/v1/subscription/checkout/wechat  — create WeChat Pay order
  POST /api/v1/subscription/webhook/stripe   — Stripe payment webhook
  POST /api/v1/subscription/webhook/wechat   — WeChat payment webhook
  POST /api/v1/subscription/cancel           — cancel Stripe subscription
  POST /api/v1/subscription/purchase-overage — purchase extra run/report credits

Adapted from CompareGPT's payment operate_view.py.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from pwm_platform.auth.dependencies import get_current_user
from pwm_platform.config import settings
from pwm_platform.db.database import get_db
from pwm_platform.db.models import User
from pwm_platform.services.billing_service import (
    ADDON_PRICES,
    BillingService,
    OVERAGE_REPORT_PRICES,
    OVERAGE_RUN_PRICES,
    PLANS,
    WECHAT_CREDIT_PACKS,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/subscription", tags=["subscription"])


# ── Stripe price map (plan_tier + period → Stripe Price ID) ──────────

def _stripe_price_id(plan_tier: str, period: str) -> Optional[str]:
    """Map plan+period to the Stripe Price ID from settings."""
    mapping = {
        ("researcher", "monthly"): settings.STRIPE_PRICE_RESEARCHER_MONTHLY,
        ("researcher", "yearly"): settings.STRIPE_PRICE_RESEARCHER_YEARLY,
        ("pro", "monthly"): settings.STRIPE_PRICE_PRO_MONTHLY,
        ("pro", "yearly"): settings.STRIPE_PRICE_PRO_YEARLY,
        ("team", "monthly"): settings.STRIPE_PRICE_TEAM_MONTHLY,
        ("team", "yearly"): settings.STRIPE_PRICE_TEAM_YEARLY,
    }
    return mapping.get((plan_tier, period)) or None


# ═════════════════════════════════════════════════════════════════════════
#  Read-only endpoints
# ═════════════════════════════════════════════════════════════════════════


@router.get("/balance")
async def get_balance(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the authenticated user's credit balance and plan info."""
    svc = BillingService(db)
    balance = await svc.get_account_balance(user.id)
    plan = svc.get_plan_info(balance["plan_tier"])
    return {
        "success": True,
        "balance": balance,
        "plan": {
            "tier": balance["plan_tier"],
            "display_name": plan["display_name"],
            "features": plan["features"],
            "max_dataset_size": plan.get("max_dataset_size", "small"),
        },
    }


@router.get("/transactions")
async def get_transactions(
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the user's credit transaction history."""
    svc = BillingService(db)
    txns = await svc.get_transaction_history(user.id, limit=limit, offset=offset)
    return {"success": True, "transactions": txns}


@router.get("/payments")
async def get_payments(
    limit: int = Query(50, le=200),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the user's payment order history."""
    svc = BillingService(db)
    payments = await svc.get_payment_history(user.id, limit=limit)
    return {"success": True, "payments": payments}


@router.get("/plans")
async def get_plans():
    """Return all available subscription plans (public, no auth required)."""
    return {
        "success": True,
        "plans": {k: {kk: vv for kk, vv in v.items()} for k, v in PLANS.items()},
        "wechat_packs": WECHAT_CREDIT_PACKS,
        "overage_run_prices": OVERAGE_RUN_PRICES,
        "overage_report_prices": OVERAGE_REPORT_PRICES,
        "addon_prices": ADDON_PRICES,
    }


# ═════════════════════════════════════════════════════════════════════════
#  Stripe Checkout
# ═════════════════════════════════════════════════════════════════════════


@router.post("/checkout/stripe")
async def create_stripe_checkout(
    plan_tier: str = Query(..., description="Plan tier: researcher, pro, team"),
    period: str = Query("monthly", description="monthly or yearly"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a Stripe Checkout Session and return the URL.
    Frontend redirects the user to this URL to complete payment.
    """
    if not settings.STRIPE_API_KEY:
        raise HTTPException(503, "Stripe payments not configured")

    if plan_tier not in ("researcher", "pro", "team"):
        raise HTTPException(400, f"Invalid plan tier: {plan_tier}")
    if period not in ("monthly", "yearly"):
        raise HTTPException(400, f"Invalid period: {period}")

    price_id = _stripe_price_id(plan_tier, period)
    if not price_id:
        raise HTTPException(400, f"Stripe price not configured for {plan_tier}/{period}")

    plan = PLANS[plan_tier]
    price_usd = plan["price_monthly"] if period == "monthly" else plan["price_yearly"]

    import stripe
    stripe.api_key = settings.STRIPE_API_KEY

    # Create order record first
    svc = BillingService(db)
    order = await svc.create_order(
        user_id=user.id,
        order_type="subscription",
        plan_tier=plan_tier,
        amount_usd=price_usd,
        payment_method="stripe",
    )

    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            mode="subscription",
            metadata={
                "orderId": order.order_id,
                "userId": str(user.id),
                "planTier": plan_tier,
                "period": period,
            },
            success_url=settings.STRIPE_SUCCESS_URL,
            cancel_url=settings.STRIPE_CANCEL_URL,
            customer_email=user.email,
        )
    except stripe.StripeError as e:
        logger.error("Stripe checkout error: %s", e)
        raise HTTPException(502, f"Payment provider error: {e.user_message}")

    return {
        "success": True,
        "checkout_url": session.url,
        "session_id": session.id,
        "order_id": order.order_id,
    }


@router.post("/webhook/stripe")
async def stripe_webhook(
    request: Request,
    stripe_signature: Optional[str] = Header(None, alias="stripe-signature"),
    db: AsyncSession = Depends(get_db),
):
    """
    Stripe webhook — handles checkout.session.completed events.
    Adapted from CompareGPT's webhook handler.
    """
    if not settings.STRIPE_API_KEY or not settings.STRIPE_WEBHOOK_SECRET:
        raise HTTPException(503, "Stripe not configured")

    import stripe
    stripe.api_key = settings.STRIPE_API_KEY

    payload = await request.body()

    try:
        event = stripe.Webhook.construct_event(
            payload, stripe_signature, settings.STRIPE_WEBHOOK_SECRET
        )
    except (ValueError, stripe.SignatureVerificationError) as e:
        logger.warning("Stripe webhook signature failed: %s", e)
        raise HTTPException(400, "Invalid signature")

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        metadata = session.get("metadata", {})
        order_id = metadata.get("orderId")
        subscription_id = session.get("subscription", "")

        if order_id:
            svc = BillingService(db)
            await svc.complete_order(order_id, payment_ref=subscription_id or session["id"])
            logger.info("Stripe checkout completed: order=%s", order_id)

    elif event["type"] == "customer.subscription.deleted":
        # Subscription cancelled or expired
        sub = event["data"]["object"]
        sub_id = sub["id"]
        logger.info("Stripe subscription deleted: %s", sub_id)
        # Could downgrade user here if needed

    return JSONResponse({"received": True})


# ═════════════════════════════════════════════════════════════════════════
#  WeChat Pay (one-time credit packs)
# ═════════════════════════════════════════════════════════════════════════


@router.post("/checkout/wechat")
async def create_wechat_checkout(
    pack_key: str = Query(..., description="Credit pack key, e.g. researcher_month"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a WeChat Pay order for a credit pack (one-time payment).

    Since WeChat Pay only supports one-time payments, we sell "Credit Packs"
    (充值包) that provide the same credits as a subscription tier for a fixed
    validity period.  This is the standard Chinese market model — similar to
    how Bilibili, iQiyi, and Tencent Video sell VIP月卡/季卡/年卡.

    Returns a QR code URL or payment link that the user scans to pay.
    """
    if not settings.WECHAT_MCHID:
        raise HTTPException(503, "WeChat Pay not configured")

    pack = WECHAT_CREDIT_PACKS.get(pack_key)
    if not pack:
        raise HTTPException(400, f"Invalid credit pack: {pack_key}")

    svc = BillingService(db)
    order = await svc.create_order(
        user_id=user.id,
        order_type="credit_pack",
        plan_tier=pack["plan_tier"],
        amount_usd=pack["price_usd"],
        payment_method="wechat",
        pack_key=pack_key,
    )

    # Update CNY amount on order
    from sqlalchemy import update as sql_update
    from pwm_platform.db.models import PaymentOrder
    await db.execute(
        sql_update(PaymentOrder)
        .where(PaymentOrder.order_id == order.order_id)
        .values(amount_cny=pack["price_cny"])
    )
    await db.commit()

    try:
        from wechatpayv3 import WeChatPay, WeChatPayType

        wxpay = WeChatPay(
            wechatpay_type=WeChatPayType.NATIVE,
            mchid=settings.WECHAT_MCHID,
            private_key=settings.WECHAT_PRIVATE_KEY,
            cert_serial_no=settings.WECHAT_CERT_SERIAL_NO,
            apiv3_key=settings.WECHAT_APIV3_KEY,
            appid=settings.WECHAT_APPID,
            notify_url=settings.WECHAT_NOTIFY_URL,
            public_key=settings.WECHAT_PUBLIC_KEY,
            public_key_id=settings.WECHAT_PUBLIC_KEY_ID,
        )

        amount_fen = int(pack["price_cny"] * 100)  # CNY → fen (分)

        code, result = wxpay.pay(
            description=f"PWM {pack['display_name']}",
            out_trade_no=order.order_id,
            amount={"total": amount_fen, "currency": "CNY"},
        )

        if code == 200:
            result_data = json.loads(result) if isinstance(result, str) else result
            return {
                "success": True,
                "order_id": order.order_id,
                "qr_code_url": result_data.get("code_url", ""),
                "pack": {
                    "name": pack["display_name"],
                    "name_zh": pack["display_name_zh"],
                    "price_cny": pack["price_cny"],
                    "run_credits": pack["run_credits"],
                    "report_credits": pack["report_credits"],
                    "validity_days": pack["validity_days"],
                },
            }
        else:
            logger.error("WeChat Pay error: code=%s result=%s", code, result)
            raise HTTPException(502, "WeChat Pay order creation failed")

    except ImportError:
        # wechatpayv3 not installed — return order info for manual processing
        logger.warning("wechatpayv3 not installed, returning order for manual processing")
        return {
            "success": True,
            "order_id": order.order_id,
            "manual_payment": True,
            "pack": {
                "name": pack["display_name"],
                "name_zh": pack["display_name_zh"],
                "price_cny": pack["price_cny"],
                "run_credits": pack["run_credits"],
                "report_credits": pack["report_credits"],
                "validity_days": pack["validity_days"],
            },
            "message": "WeChat Pay SDK not available. Contact support for manual payment.",
        }
    except Exception as e:
        logger.error("WeChat Pay error: %s", e)
        raise HTTPException(502, f"WeChat Pay error: {e}")


@router.post("/webhook/wechat")
async def wechat_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    WeChat Pay callback — handles payment completion notifications.
    Adapted from CompareGPT's wechatwebhook handler.
    """
    if not settings.WECHAT_MCHID:
        raise HTTPException(503, "WeChat Pay not configured")

    try:
        body = await request.body()
        data = json.loads(body)

        # Decrypt the resource field using APIV3 key
        resource = data.get("resource", {})
        # In production, you'd decrypt resource.ciphertext with APIV3_KEY
        # For now, handle the out_trade_no from the notification
        out_trade_no = None

        if "ciphertext" in resource:
            try:
                from wechatpayv3 import WeChatPay, WeChatPayType
                wxpay = WeChatPay(
                    wechatpay_type=WeChatPayType.NATIVE,
                    mchid=settings.WECHAT_MCHID,
                    private_key=settings.WECHAT_PRIVATE_KEY,
                    cert_serial_no=settings.WECHAT_CERT_SERIAL_NO,
                    apiv3_key=settings.WECHAT_APIV3_KEY,
                    appid=settings.WECHAT_APPID,
                    notify_url=settings.WECHAT_NOTIFY_URL,
                    public_key=settings.WECHAT_PUBLIC_KEY,
                    public_key_id=settings.WECHAT_PUBLIC_KEY_ID,
                )
                decrypted = wxpay.decrypt_callback(
                    resource["nonce"],
                    resource["ciphertext"],
                    resource["associated_data"],
                )
                decrypted_data = json.loads(decrypted) if isinstance(decrypted, str) else decrypted
                out_trade_no = decrypted_data.get("out_trade_no")
                trade_state = decrypted_data.get("trade_state")
            except Exception as e:
                logger.error("WeChat decrypt error: %s", e)
                raise HTTPException(400, "Failed to decrypt notification")
        else:
            out_trade_no = data.get("out_trade_no")
            trade_state = data.get("trade_state", "SUCCESS")

        if out_trade_no and trade_state == "SUCCESS":
            svc = BillingService(db)
            await svc.complete_order(out_trade_no, payment_ref=f"wechat_{out_trade_no}")
            logger.info("WeChat payment completed: order=%s", out_trade_no)

        return JSONResponse({"code": "SUCCESS", "message": "OK"})

    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid request body")


# ═════════════════════════════════════════════════════════════════════════
#  Subscription management
# ═════════════════════════════════════════════════════════════════════════


@router.post("/cancel")
async def cancel_subscription(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Cancel the user's active Stripe subscription."""
    svc = BillingService(db)
    result = await svc.cancel_subscription(user.id)

    if "error" in result:
        raise HTTPException(400, result["error"])

    # Cancel on Stripe side
    sub_id = result.get("subscription_id")
    if sub_id and settings.STRIPE_API_KEY:
        try:
            import stripe
            stripe.api_key = settings.STRIPE_API_KEY
            stripe.Subscription.modify(sub_id, cancel_at_period_end=True)
        except Exception as e:
            logger.error("Stripe cancel error: %s", e)

    return {"success": True, "message": "Subscription will be cancelled at end of billing period."}


# ═════════════════════════════════════════════════════════════════════════
#  Overage purchases
# ═════════════════════════════════════════════════════════════════════════


@router.post("/purchase-overage")
async def purchase_overage(
    credit_kind: str = Query(..., description="run or report"),
    size: str = Query("small", description="small/medium/large for runs; standard/advanced for reports"),
    quantity: int = Query(1, ge=1, le=100),
    payment_method: str = Query("stripe", description="stripe or wechat"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Purchase additional run or report credits (overage / add-on).
    Returns a checkout URL for the selected payment method.
    """
    if credit_kind == "run":
        unit_price = OVERAGE_RUN_PRICES.get(size, 3.0)
    elif credit_kind == "report":
        unit_price = OVERAGE_REPORT_PRICES.get(size, 25.0)
    else:
        raise HTTPException(400, "credit_kind must be 'run' or 'report'")

    total = unit_price * quantity
    order_type = f"overage_{credit_kind}"

    svc = BillingService(db)
    order = await svc.create_order(
        user_id=user.id,
        order_type=order_type,
        plan_tier="addon",
        amount_usd=total,
        payment_method=payment_method,
    )

    # Store credit amount on order for provisioning
    from sqlalchemy import update as sql_update
    from pwm_platform.db.models import PaymentOrder
    await db.execute(
        sql_update(PaymentOrder)
        .where(PaymentOrder.order_id == order.order_id)
        .values(credit_amount=quantity)
    )
    await db.commit()

    if payment_method == "stripe" and settings.STRIPE_API_KEY:
        import stripe
        stripe.api_key = settings.STRIPE_API_KEY

        try:
            session = stripe.checkout.Session.create(
                payment_method_types=["card"],
                line_items=[{
                    "price_data": {
                        "currency": "usd",
                        "product_data": {
                            "name": f"PWM {credit_kind.title()} Credits ({size}) x{quantity}",
                        },
                        "unit_amount": int(unit_price * 100),
                    },
                    "quantity": quantity,
                }],
                mode="payment",
                metadata={
                    "orderId": order.order_id,
                    "userId": str(user.id),
                    "creditKind": credit_kind,
                    "quantity": str(quantity),
                },
                success_url=settings.STRIPE_SUCCESS_URL,
                cancel_url=settings.STRIPE_CANCEL_URL,
                customer_email=user.email,
            )
            return {
                "success": True,
                "checkout_url": session.url,
                "order_id": order.order_id,
                "total_usd": total,
            }
        except Exception as e:
            logger.error("Stripe overage checkout error: %s", e)
            raise HTTPException(502, f"Payment error: {e}")

    return {
        "success": True,
        "order_id": order.order_id,
        "total_usd": total,
        "message": "Order created. Use payment link to complete.",
    }
