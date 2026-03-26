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

from fastapi import APIRouter, Depends, Form, Header, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession

from pwm_platform.auth.dependencies import get_current_user
from pwm_platform.auth.service import auth_service
from pwm_platform.config import settings
from pwm_platform.db.database import get_db
from pwm_platform.db.models import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])

templates = Jinja2Templates(directory="pwm_platform/templates")

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


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Log out the current user."""
    await auth_service.logout_user(user.id, db)

    # Browser form submission → redirect to homepage
    content_type = request.headers.get("content-type", "")
    accept = request.headers.get("accept", "")
    if "text/html" in accept or "application/x-www-form-urlencoded" in content_type:
        redirect = RedirectResponse("/benchmark", status_code=303)
        redirect.delete_cookie("access_token", path="/")
        return redirect

    response.delete_cookie("access_token", path="/")
    return {"success": True, "message": "Logged out successfully"}


@router.post("/signup")
async def signup(
    request: Request,
    response: Response,
    email: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    """HTMX signup form handler. Returns HX-Redirect on success, HTML error on failure."""
    if len(password) < 8:
        return templates.TemplateResponse(request, "signup.html", {
            "error": "Password must be at least 8 characters",
            "email": email,
            "username": username,
            "google_client_id": settings.GOOGLE_CLIENT_ID,
        }, status_code=400)

    try:
        result = await auth_service.create_local_user(email, username, password, db)
    except HTTPException as exc:
        error = exc.detail if isinstance(exc.detail, str) else "Registration failed"
        return templates.TemplateResponse(request, "signup.html", {
            "error": error,
            "email": email,
            "username": username,
            "google_client_id": settings.GOOGLE_CLIENT_ID,
        }, status_code=400)

    response.set_cookie(value=result["access_token"], **_COOKIE_KWARGS)
    response.headers["HX-Redirect"] = "/benchmark"
    return response


@router.post("/signup-form")
async def signup_form(
    request: Request,
    email: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    """Handle registration from HTML form. Sets cookie + redirects to /benchmark."""
    if len(password) < 8:
        return templates.TemplateResponse(request, "signup.html", {
            "error": "Password must be at least 8 characters",
            "email": email,
            "username": username,
            "google_client_id": settings.GOOGLE_CLIENT_ID,
        }, status_code=400)

    try:
        result = await auth_service.create_local_user(email, username, password, db)
    except HTTPException as exc:
        error = exc.detail if isinstance(exc.detail, str) else "Registration failed"
        return templates.TemplateResponse(request, "signup.html", {
            "error": error,
            "email": email,
            "username": username,
            "google_client_id": settings.GOOGLE_CLIENT_ID,
        }, status_code=400)

    redirect = RedirectResponse("/benchmark", status_code=303)
    redirect.set_cookie(value=result["access_token"], **_COOKIE_KWARGS)
    return redirect


@router.post("/forgot-password-form")
async def forgot_password_form(
    request: Request,
    email: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    """Send password reset email (form submission)."""
    try:
        await auth_service.request_password_reset(email, db)
    except Exception as exc:
        logger.error("Password reset error: %s", exc)
        # Still show success to avoid user enumeration
    return templates.TemplateResponse(request, "forgot_password.html", {
        "success": True,
        "email": email,
    })


@router.post("/reset-password-form")
async def reset_password_form(
    request: Request,
    token: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    """Process password reset with token (form submission)."""
    try:
        await auth_service.confirm_password_reset(token, password, db)
    except HTTPException as exc:
        error = exc.detail if isinstance(exc.detail, str) else "Reset failed"
        return templates.TemplateResponse(request, "reset_password.html", {
            "error": error,
            "token": token,
        }, status_code=400)

    return RedirectResponse("/login?reset=1", status_code=303)


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
