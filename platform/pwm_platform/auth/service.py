"""
Authentication Service — adapted from CompareGPT-AIScientist ``auth_service.py``.

Supports two auth flows:
  A) SSO token exchange  (CompareGPT-compatible)
  B) Local email + password login  (PWM addition)

Both flows result in a JWT access token set as an HttpOnly cookie.
"""

from __future__ import annotations

import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from pwm_platform.auth.passwords import hash_password, verify_password
from pwm_platform.auth.token_manager import get_token_manager
from pwm_platform.config import settings
from pwm_platform.db.models import PasswordResetToken, User
from pwm_platform.services.pwm_token_service import pwm_token_service

logger = logging.getLogger(__name__)


class AuthService:
    """Stateless auth operations — each method receives its own ``db`` session."""

    # ── SSO flow (CompareGPT-compatible) ─────────────────────────────────

    async def exchange_sso_token(self, sso_token: str, db: AsyncSession) -> Dict[str, Any]:
        """Validate *sso_token* with the external SSO provider and return a JWT.

        This mirrors ``CompareGPT AuthService.exchange_sso_token()`` almost exactly.
        """
        sso_url = settings.SSO_VALIDATE_URL
        if not sso_url:
            raise HTTPException(status_code=501, detail="SSO not configured")

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    sso_url,
                    headers={"Authorization": f"Bearer {sso_token}"},
                    timeout=10.0,
                )
            except httpx.RequestError as exc:
                logger.error("SSO connection error: %s", exc)
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "sso_unavailable",
                        "message": "SSO service is currently unavailable",
                        "require_reauth": False,
                    },
                )

        if resp.status_code == 401:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "sso_token_expired",
                    "message": "SSO token expired or invalid. Please login again.",
                    "require_reauth": True,
                },
            )

        if resp.status_code != 200:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "authentication_failed",
                    "message": "SSO validation failed",
                    "require_reauth": True,
                },
            )

        data = resp.json().get("data")
        if data is None:
            data = resp.json()

        if not data.get("valid"):
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "sso_token_invalid",
                    "message": "SSO token is not valid",
                    "require_reauth": True,
                },
            )

        user_info = data.get("user_info", {})
        if not user_info:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "user_data_missing",
                    "message": "User data not found from SSO",
                    "require_reauth": True,
                },
            )

        sso_user_id = user_info.get("user_id")
        if not sso_user_id:
            raise HTTPException(status_code=401, detail="Missing user_id from SSO")

        api_key = data.get("api_key") or user_info.get("api_key")

        # Upsert user — same pattern as CompareGPT ``repo.upsert_user()``
        user = await self._upsert_sso_user(
            db,
            sso_user_id=sso_user_id,
            username=user_info.get("user_name", ""),
            role=user_info.get("role", "user"),
            sso_token=sso_token,
            api_key=api_key,
        )

        tm = get_token_manager()
        access_token = tm.create_access_token(user.id)

        return {
            "success": True,
            "access_token": access_token,
            "user": _user_to_dict(user),
        }

    # ── Local login flow (PWM addition) ──────────────────────────────────

    async def local_login(
        self, email: str, password: str, db: AsyncSession
    ) -> Dict[str, Any]:
        """Authenticate with email + password and return a JWT."""
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()

        if user is None or user.password_hash is None:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        if not verify_password(password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid email or password")

        if not user.is_active:
            raise HTTPException(status_code=403, detail="Account deactivated")

        tm = get_token_manager()
        access_token = tm.create_access_token(user.id)

        return {
            "success": True,
            "access_token": access_token,
            "user": _user_to_dict(user),
        }

    # ── Signup ───────────────────────────────────────────────────────────

    async def create_local_user(
        self,
        email: str,
        username: str,
        password: str,
        db: AsyncSession,
    ) -> Dict[str, Any]:
        """Create a new local user with hashed password."""
        # Check for existing email
        result = await db.execute(select(User).where(User.email == email))
        if result.scalar_one_or_none() is not None:
            raise HTTPException(status_code=409, detail="Email already registered")

        user = User(
            email=email,
            username=username,
            password_hash=hash_password(password),
            role="user",
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)

        await pwm_token_service.provision_pwm_account(user.id, db)

        tm = get_token_manager()
        access_token = tm.create_access_token(user.id)

        logger.info("Created local user: %s (id=%s)", email, user.id)
        return {
            "success": True,
            "access_token": access_token,
            "user": _user_to_dict(user),
        }

    # ── Google OAuth login ───────────────────────────────────────────────

    async def google_login(
        self, credential: str, db: AsyncSession
    ) -> Dict[str, Any]:
        """Verify a Google Identity Services credential and return a JWT.

        Uses Google's tokeninfo endpoint to verify the credential without
        requiring the google-auth library directly.
        """
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(
                    "https://oauth2.googleapis.com/tokeninfo",
                    params={"id_token": credential},
                    timeout=10.0,
                )
            except httpx.RequestError as exc:
                logger.error("Google tokeninfo request error: %s", exc)
                raise HTTPException(status_code=503, detail="Google auth service unavailable")

        if resp.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid Google credential")

        info = resp.json()

        # Verify audience matches our client ID
        if settings.GOOGLE_CLIENT_ID and info.get("aud") != settings.GOOGLE_CLIENT_ID:
            raise HTTPException(status_code=401, detail="Google credential audience mismatch")

        email = info.get("email", "")
        if not email or not info.get("email_verified") == "true":
            raise HTTPException(status_code=401, detail="Google email not verified")

        google_sub = info.get("sub", "")
        name = info.get("name") or info.get("given_name") or email.split("@")[0]

        # Find existing user by Google sub or email, or create new
        from sqlalchemy import select as _select
        result = await db.execute(
            _select(User).where(
                (User.sso_user_id == int(google_sub[:9])) |
                (User.email == email)
            )
        )
        user = result.scalar_one_or_none()

        is_new_user = user is None
        if user is None:
            user = User(
                email=email,
                username=name,
                role="user",
                sso_user_id=int(google_sub[:9]) if google_sub.isdigit() else None,
            )
            db.add(user)
        else:
            # Update name if changed
            if name and not user.password_hash:
                user.username = name

        await db.commit()
        await db.refresh(user)

        if is_new_user:
            await pwm_token_service.provision_pwm_account(user.id, db)

        tm = get_token_manager()
        access_token = tm.create_access_token(user.id)

        logger.info("Google login: %s (id=%s)", email, user.id)
        return {
            "success": True,
            "access_token": access_token,
            "user": _user_to_dict(user),
        }

    # ── Token validation (CompareGPT /validate mode 2) ───────────────────

    async def validate_access_token(
        self, token: str, db: AsyncSession
    ) -> Dict[str, Any]:
        """Verify a JWT and return current user data."""
        tm = get_token_manager()
        user_id = tm.verify_access_token(token)
        if user_id is None:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "invalid_token",
                    "message": "Access token is invalid or expired",
                    "require_reauth": True,
                },
            )

        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if user is None:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "user_not_found",
                    "message": "User not found or logged out",
                    "require_reauth": True,
                },
            )

        return {
            "success": True,
            "valid": True,
            "user": _user_to_dict(user),
        }

    # ── SIWE (Sign-In with Ethereum) ─────────────────────────────────────

    async def login_or_create_siwe_user(
        self,
        wallet_address: str,
        db: AsyncSession,
    ) -> Dict[str, Any]:
        """Look up a user by wallet_address, creating one if absent.

        The display username defaults to the wallet's short form (first 6 +
        last 4 hex chars). Users can change their display name later from
        their profile page.
        """
        addr = wallet_address.lower()
        result = await db.execute(select(User).where(User.wallet_address == addr))
        user = result.scalar_one_or_none()

        if user is None:
            short = f"{addr[:6]}…{addr[-4:]}"
            user = User(
                username=short,
                wallet_address=addr,
                auth_method="siwe",
                role="user",
                email=None,
                password_hash=None,
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)
            # SIWE users already have a wallet — register it directly, no custodial key
            await pwm_token_service.provision_pwm_account(user.id, db, existing_wallet=addr)
            logger.info("Created SIWE user %s (id=%s)", addr, user.id)
        else:
            # Update last-login marker (auth_method could be local/sso already
            # if the user previously linked an address; preserve original).
            if user.auth_method == "local":
                # First time signing in via wallet on an existing local account:
                # mark the auth method as hybrid by keeping local + filling
                # wallet_address. Nothing to flip.
                pass

        if not user.is_active:
            raise HTTPException(status_code=403, detail="Account is deactivated")

        tm = get_token_manager()
        access_token = tm.create_access_token(user.id)
        return {
            "success": True,
            "access_token": access_token,
            "user": _user_to_dict(user),
        }

    # ── Logout ───────────────────────────────────────────────────────────

    async def logout_user(self, user_id: int, db: AsyncSession) -> Dict[str, Any]:
        """Clear SSO token on logout (mirrors CompareGPT ``clear_user_data``)."""
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if user:
            user.sso_token = None
            await db.commit()
            logger.info("Cleared auth data for user %s", user_id)

        return {"success": True, "message": "Logged out successfully"}

    # ── API key management ────────────────────────────────────────────────────

    async def generate_api_key(self, user_id: int, db: AsyncSession) -> str:
        """Generate a new personal API key, replacing any existing one."""
        new_key = "pwm_" + secrets.token_urlsafe(32)

        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")

        user.api_key = new_key
        await db.commit()
        logger.info("Generated new API key for user %s", user_id)
        return new_key

    async def revoke_api_key(self, user_id: int, db: AsyncSession) -> None:
        """Revoke the user's API key."""
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if user:
            user.api_key = None
            await db.commit()
            logger.info("Revoked API key for user %s", user_id)

    # ── Password reset ───────────────────────────────────────────────────

    async def request_password_reset(self, email: str, db: AsyncSession) -> None:
        """Generate a reset token and send a password reset email.

        Silently succeeds even if email not found (to avoid user enumeration).
        """
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        if user is None or user.password_hash is None:
            return  # silent — don't reveal whether email exists

        token = secrets.token_urlsafe(48)
        expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        row = PasswordResetToken(user_id=user.id, token=token, expires_at=expires_at)
        db.add(row)
        await db.commit()

        from pwm_platform.auth.email import send_password_reset_email
        await send_password_reset_email(user.email, user.username, token)
        logger.info("Password reset email sent to %s", email)

    async def confirm_password_reset(
        self, token: str, new_password: str, db: AsyncSession
    ) -> None:
        """Validate the token and update the user's password."""
        if len(new_password) < 8:
            raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

        result = await db.execute(
            select(PasswordResetToken).where(PasswordResetToken.token == token)
        )
        row = result.scalar_one_or_none()

        now = datetime.now(timezone.utc)
        if row is None or row.used or row.expires_at < now:
            raise HTTPException(status_code=400, detail="Reset link is invalid or has expired")

        # Mark token used
        row.used = True

        # Update password
        user_result = await db.execute(select(User).where(User.id == row.user_id))
        user = user_result.scalar_one_or_none()
        if user is None:
            raise HTTPException(status_code=400, detail="User not found")

        user.password_hash = hash_password(new_password)
        await db.commit()
        logger.info("Password reset completed for user %s", user.email)

    # ── Internal helpers ─────────────────────────────────────────────────

    async def _upsert_sso_user(
        self,
        db: AsyncSession,
        *,
        sso_user_id: int,
        username: str,
        role: str,
        sso_token: str,
        api_key: Optional[str],
    ) -> User:
        """Insert or update an SSO-authenticated user."""
        result = await db.execute(
            select(User).where(User.sso_user_id == sso_user_id)
        )
        user = result.scalar_one_or_none()

        is_new_sso_user = user is None
        if user:
            if username:
                user.username = username
            if role:
                user.role = role
            user.sso_token = sso_token
            if api_key is not None:
                user.api_key = api_key
        else:
            user = User(
                sso_user_id=sso_user_id,
                username=username or f"user_{sso_user_id}",
                role=role or "user",
                sso_token=sso_token,
                api_key=api_key,
            )
            db.add(user)

        await db.commit()
        await db.refresh(user)

        if is_new_sso_user:
            await pwm_token_service.provision_pwm_account(user.id, db)
        return user


# ── Module-level singleton ───────────────────────────────────────────────

auth_service = AuthService()


# ── Serialisation helper ─────────────────────────────────────────────────


def _user_to_dict(user: User) -> dict:
    """Serialize a User row to a CompareGPT-compatible dict."""
    return {
        "user_info": {
            "user_id": user.id,
            "user_name": user.username,
            "role": user.role,
            "wallet_address": user.wallet_address,
            "auth_method": user.auth_method,
        },
        "balance": {
            "credit": 0,
            "token": 0,
        },
        "api_key": user.api_key or "",
    }
