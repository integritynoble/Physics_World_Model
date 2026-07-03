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

from pwm_platform.auth.csrf import enforce_csrf
from pwm_platform.auth.dependencies import get_current_user
from pwm_platform.auth.service import auth_service
from pwm_platform.auth.token_manager import get_token_manager
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
    csrf_token: str = Form(""),
    db: AsyncSession = Depends(get_db),
):
    """Handle registration from HTML form. Sets cookie + redirects to /benchmark."""
    enforce_csrf(request, csrf_token)
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


@router.post("/login-form")
async def login_form(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    next: str = Form(default="/benchmark"),
    csrf_token: str = Form(""),
    db: AsyncSession = Depends(get_db),
):
    """Handle login from HTML form. Sets cookie + redirects on success."""
    enforce_csrf(request, csrf_token)
    try:
        result = await auth_service.local_login(email, password, db)
    except HTTPException:
        return templates.TemplateResponse(request, "login.html", {
            "error": "Invalid email or password.",
            "sso_enabled": bool(settings.SSO_REDIRECT_URL),
            "sso_url": settings.SSO_REDIRECT_URL,
            "google_client_id": settings.GOOGLE_CLIENT_ID,
        }, status_code=401)

    redirect_to = _safe_next(next)
    redirect = RedirectResponse(redirect_to, status_code=303)
    redirect.set_cookie(value=result["access_token"], **_COOKIE_KWARGS)
    return redirect


def _safe_next(next_url: str) -> str:
    """Return a safe relative redirect path, blocking open-redirect attacks.

    Only paths starting with a single '/' (not '//' or '/\\') are allowed.
    """
    if not next_url or not next_url.startswith("/"):
        return "/benchmark"
    # Block protocol-relative URLs like "//evil.com" or "/\\evil.com"
    if next_url.startswith("//") or next_url.startswith("/\\"):
        return "/benchmark"
    return next_url


@router.post("/google")
async def google_login(
    credential: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    """Verify a Google Sign-In credential (ID token) and return a JWT."""
    if not settings.GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=501, detail="Google OAuth not configured")

    try:
        from google.oauth2 import id_token
        from google.auth.transport import requests as google_requests

        id_info = id_token.verify_oauth2_token(
            credential,
            google_requests.Request(),
            settings.GOOGLE_CLIENT_ID,
        )
    except ValueError as exc:
        logger.warning("Google token verification failed: %s", exc)
        raise HTTPException(status_code=401, detail="Invalid Google credential")
    except Exception as exc:
        logger.exception("Unexpected error verifying Google token: %s", exc)
        raise HTTPException(status_code=500, detail="Google sign-in temporarily unavailable")

    email = id_info.get("email")
    name = id_info.get("name") or id_info.get("given_name") or ""
    google_sub = id_info.get("sub")
    email_verified = id_info.get("email_verified", False)

    if not email or not google_sub:
        raise HTTPException(status_code=401, detail="Missing fields in Google token")
    if not email_verified:
        raise HTTPException(status_code=401, detail="Google account email is not verified")

    user = await auth_service.upsert_google_user(db, google_sub=google_sub, email=email, name=name)

    tm = get_token_manager()
    access_token = tm.create_access_token(user.id)

    redirect = RedirectResponse("/benchmark", status_code=303)
    redirect.set_cookie(value=access_token, **_COOKIE_KWARGS)
    return redirect


# ── API key management ───────────────────────────────────────────────────


@router.get("/api-key")
async def get_api_key(user: User = Depends(get_current_user)):
    """Return the current user's API key (masked except last 4 chars)."""
    key = user.api_key or ""
    if key:
        masked = key[:8] + "..." + key[-4:]
    else:
        masked = None
    return {"success": True, "has_key": bool(key), "masked_key": masked}


@router.post("/api-key/generate")
async def generate_api_key(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate a new API key, revoking any existing one. Returns the full key once."""
    new_key = await auth_service.generate_api_key(user.id, db)
    return {
        "success": True,
        "api_key": new_key,
        "note": "Save this key now — it will not be shown again in full.",
    }


@router.delete("/api-key")
async def revoke_api_key(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Revoke the current API key."""
    await auth_service.revoke_api_key(user.id, db)
    return {"success": True, "message": "API key revoked"}


class DeleteAccountBody(BaseModel):
    confirm: str


@router.post("/delete-account")
async def delete_account(
    body: DeleteAccountBody,
    request: Request,
    response: Response,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Permanently deactivate + anonymize the logged-in account.

    The PWM ledger rows and custodial wallet stay in the database for audit
    integrity, but the account can no longer log in, its email is freed for
    a fresh registration, and the API key is revoked. Irreversible.
    """
    enforce_csrf(request)
    if body.confirm != "DELETE":
        raise HTTPException(status_code=400, detail='Type "DELETE" to confirm')

    user.is_active = False
    user.email = f"deleted-{user.id}@deleted.invalid"
    user.username = "Deleted user"
    user.password_hash = None
    user.sso_user_id = None
    user.sso_token = None
    user.api_key = None
    await db.commit()

    response.delete_cookie("access_token", path="/")
    return {"success": True, "message": "Account deleted"}


@router.post("/forgot-password-form")
async def forgot_password_form(
    request: Request,
    email: str = Form(...),
    csrf_token: str = Form(""),
    db: AsyncSession = Depends(get_db),
):
    """Send password reset email (form submission)."""
    enforce_csrf(request, csrf_token)
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
    csrf_token: str = Form(""),
    db: AsyncSession = Depends(get_db),
):
    """Process password reset with token (form submission)."""
    enforce_csrf(request, csrf_token)
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
