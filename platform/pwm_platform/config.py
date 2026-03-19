"""
PWM Platform Configuration
Pydantic Settings — all config from environment variables / .env file.

SECURITY: No secrets have default values. All must be provided via .env or env vars.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve paths relative to the platform/ package root
_PLATFORM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Settings(BaseSettings):
    """Application settings — loaded from environment / .env file."""

    # ── General ──────────────────────────────────────────────────────────
    APP_NAME: str = "PWM Platform"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    APP_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # ── Security / Auth ──────────────────────────────────────────────────
    SECRET_KEY: str                          # REQUIRED — no default
    ACCESS_TOKEN_EXPIRE_DAYS: int = 7
    BCRYPT_ROUNDS: int = 12

    # CSRF protection for form-based UI
    CSRF_SECRET: str                         # REQUIRED — no default

    # Rate limiting (requests per minute on auth endpoints)
    AUTH_RATE_LIMIT: int = 10

    # ── SSO (CompareGPT-compatible) ──────────────────────────────────────
    SSO_VALIDATE_URL: str = ""               # e.g. https://auth.comparegpt.io/api/sso/validate
    SSO_REDIRECT_URL: str = ""               # e.g. https://auth.comparegpt.io/sso-redirect?redirect=...

    # ── Database ─────────────────────────────────────────────────────────
    DATABASE_URL: str                        # e.g. postgresql+asyncpg://user:pass@localhost/pwm

    # ── Redis ────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379"

    # ── CORS ─────────────────────────────────────────────────────────────
    CORS_ORIGINS: List[str] = ["https://pwm.platformai.org"]

    # ── Modal (GPU workers) ──────────────────────────────────────────────
    MODAL_TOKEN_ID: str = ""
    MODAL_TOKEN_SECRET: str = ""

    # ── GCP ──────────────────────────────────────────────────────────────
    GCS_BUCKET: str = "pwm-benchmark-datasets"
    GOOGLE_APPLICATION_CREDENTIALS: str = ""
    BIGQUERY_PROJECT: str = ""
    BIGQUERY_DATASET: str = "pwm_analytics"

    # ── Stripe (credit card payments) ─────────────────────────────────
    STRIPE_API_KEY: str = ""                     # sk_live_... or sk_test_...
    STRIPE_WEBHOOK_SECRET: str = ""              # whsec_...
    STRIPE_SUCCESS_URL: str = "https://pwm.platformai.org/subscription?status=success"
    STRIPE_CANCEL_URL: str = "https://pwm.platformai.org/pricing?status=cancelled"

    # Stripe Price IDs for each plan (created in Stripe Dashboard)
    STRIPE_PRICE_RESEARCHER_MONTHLY: str = ""    # price_...
    STRIPE_PRICE_RESEARCHER_YEARLY: str = ""
    STRIPE_PRICE_PRO_MONTHLY: str = ""
    STRIPE_PRICE_PRO_YEARLY: str = ""
    STRIPE_PRICE_TEAM_MONTHLY: str = ""
    STRIPE_PRICE_TEAM_YEARLY: str = ""

    # ── WeChat Pay (one-time credit packs) ────────────────────────────
    WECHAT_MCHID: str = ""                       # merchant ID
    WECHAT_APPID: str = ""
    WECHAT_APIV3_KEY: str = ""
    WECHAT_PRIVATE_KEY: str = ""                  # RSA private key (PEM)
    WECHAT_CERT_SERIAL_NO: str = ""
    WECHAT_PUBLIC_KEY: str = ""
    WECHAT_PUBLIC_KEY_ID: str = ""
    WECHAT_NOTIFY_URL: str = "https://pwm.platformai.org/api/v1/subscription/webhook/wechat"
    WECHAT_PAY_EXCHANGE_RATE: float = 7.1        # USD→CNY fallback rate

    # ── Pydantic-settings config ─────────────────────────────────────────
    model_config = SettingsConfigDict(
        env_file=os.path.join(_PLATFORM_DIR, ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
