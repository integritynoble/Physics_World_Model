"""
Auth Router — CompareGPT-compatible SSO + local login.

Endpoints:
    POST /api/v1/auth/validate   — SSO token → JWT exchange (CompareGPT mode)
    POST /api/v1/auth/login      — local email + password login
    POST /api/v1/auth/signup     — local registration
    POST /api/v1/auth/logout     — clear session
    GET  /api/v1/auth/me         — current user info
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession

from pwm_platform.auth.dependencies import get_current_user
from pwm_platform.auth.service import auth_service
from pwm_platform.config import settings
from pwm_platform.db.database import get_db
from pwm_platform.db.models import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])

_COOKIE_KWARGS = {
    "key": "access_token",
    "httponly": True,
    "secure": True,
    "samesite": "lax",
    "max_age": 7 * 24 * 3600,
    "path": "/",
}


# ── Request / response schemas ───────────────────────────────────────────


class ValidateRequest(BaseModel):
    """CompareGPT-compatible SSO token exchange request."""
    sso_token: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


class SignupRequest(BaseModel):
    email: str
    username: str = Field(..., min_length=2, max_length=100)
    password: str = Field(..., min_length=8)


class AuthResponse(BaseModel):
    success: bool
    access_token: Optional[str] = None
    valid: Optional[bool] = None
    user: Optional[dict] = None


class LogoutResponse(BaseModel):
    success: bool
    message: str


# ── Endpoints ────────────────────────────────────────────────────────────


@router.post("/validate", response_model=AuthResponse)
async def validate(
    request_body: ValidateRequest,
    response: Response,
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db),
):
    """Unified validate — CompareGPT-compatible.

    Mode 1: ``{ "sso_token": "..." }`` → exchange for JWT
    Mode 2: ``Authorization: Bearer <token>`` → validate existing JWT
    """
    # Mode 1: SSO token exchange
    if request_body.sso_token:
        result = await auth_service.exchange_sso_token(request_body.sso_token, db)
        response.set_cookie(value=result["access_token"], **_COOKIE_KWARGS)
        return result

    # Mode 2: validate existing token
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
        return await auth_service.validate_access_token(token, db)

    raise HTTPException(
        status_code=400,
        detail={
            "error": "missing_credentials",
            "message": "Either sso_token or Authorization header is required",
            "require_reauth": True,
        },
    )


@router.post("/login", response_model=AuthResponse)
async def login(
    body: LoginRequest,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    """Local email + password login."""
    result = await auth_service.local_login(body.email, body.password, db)
    response.set_cookie(value=result["access_token"], **_COOKIE_KWARGS)
    return result


@router.post("/logout", response_model=LogoutResponse)
async def logout(
    response: Response,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Log out the current user."""
    result = await auth_service.logout_user(user.id, db)
    response.delete_cookie("access_token", path="/")
    return result


@router.get("/me", response_model=AuthResponse)
async def me(user: User = Depends(get_current_user)):
    """Return current user info."""
    return {
        "success": True,
        "valid": True,
        "user": {
            "user_info": {
                "user_id": user.id,
                "user_name": user.username,
                "role": user.role,
            },
        },
    }
