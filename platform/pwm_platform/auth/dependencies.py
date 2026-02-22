"""
FastAPI auth dependencies — reuses the CompareGPT ``Depends()`` pattern.

Two extraction modes (tried in order):
1. HttpOnly cookie ``access_token``  (server-rendered UI sessions)
2. ``Authorization: Bearer <token>`` header  (API / programmatic clients)

Security improvement over CompareGPT: tokens live in HttpOnly cookies
instead of localStorage, eliminating XSS token theft.
"""

from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from pwm_platform.auth.token_manager import get_token_manager
from pwm_platform.db.database import get_db
from pwm_platform.db.models import User


async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> User:
    """Extract and verify the authenticated user from the request.

    Raises 401 if no valid token or user not found.
    """
    token = _extract_token(request)

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

    if user is None or not user.is_active:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "user_not_found",
                "message": "User not found or deactivated",
                "require_reauth": True,
            },
        )

    return user


async def get_optional_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """Try to extract the authenticated user, return None if not logged in.

    Used for public pages that show extra content when authenticated.
    """
    token = _extract_token_or_none(request)
    if token is None:
        return None

    tm = get_token_manager()
    user_id = tm.verify_access_token(token)
    if user_id is None:
        return None

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None or not user.is_active:
        return None

    return user


def require_role(*roles: str):
    """Dependency factory for role-based access control.

    Usage::

        @router.get("/admin")
        async def admin_page(user: User = Depends(require_role("admin"))):
            ...
    """

    async def _check(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles and user.role != "admin":
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user

    return _check


# ── helpers ──────────────────────────────────────────────────────────────


def _extract_token(request: Request) -> str:
    """Try cookie first, then Authorization header. Raises 401 if missing."""
    token = _extract_token_or_none(request)
    if token is not None:
        return token

    raise HTTPException(
        status_code=401,
        detail={
            "error": "missing_token",
            "message": "Authentication required",
            "require_reauth": True,
        },
    )


def _extract_token_or_none(request: Request) -> Optional[str]:
    """Try cookie first, then Authorization header. Returns None if missing."""
    # 1. HttpOnly cookie (UI sessions)
    token = request.cookies.get("access_token")
    if token:
        return token

    # 2. Authorization header (API clients — CompareGPT compatible)
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]

    return None
