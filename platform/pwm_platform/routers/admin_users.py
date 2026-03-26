"""Admin user management -- list, edit, delete, reset passwords.

Only accessible by users with role='admin'.
Follows CompareGPT pattern: soft delete, guard against self-delete and superuser delete.
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from pwm_platform.auth.dependencies import require_role
from pwm_platform.auth.passwords import hash_password
from pwm_platform.db.database import get_db
from pwm_platform.db.models import User

router = APIRouter(prefix="/admin/users", tags=["Admin"])

_admin_required = require_role("admin")


# ── Request schemas ────────────────────────────────────────────────────


class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None  # "user", "admin", "reviewer"
    is_active: Optional[bool] = None


class PasswordReset(BaseModel):
    new_password: str


class BulkDelete(BaseModel):
    user_ids: List[int]


# ── API endpoints ──────────────────────────────────────────────────────


@router.get("/api/list")
async def list_users(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(_admin_required),
):
    """List all users with their profiles."""
    result = await db.execute(select(User).order_by(User.id))
    users = result.scalars().all()
    return {
        "users": [
            {
                "id": u.id,
                "email": u.email or "",
                "username": u.username or "",
                "role": u.role or "user",
                "is_active": u.is_active if hasattr(u, "is_active") else True,
                "created_at": str(u.created_at) if hasattr(u, "created_at") else "",
                "credit_balance": u.credit_balance if hasattr(u, "credit_balance") else 0,
            }
            for u in users
        ],
        "total": len(users),
    }


@router.get("/api/{user_id}")
async def get_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(_admin_required),
):
    """Get a single user by ID."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(404, "User not found")
    return {
        "id": user.id,
        "email": user.email or "",
        "username": user.username or "",
        "role": user.role or "user",
        "is_active": user.is_active if hasattr(user, "is_active") else True,
        "created_at": str(user.created_at) if hasattr(user, "created_at") else "",
    }


@router.put("/api/{user_id}")
async def update_user(
    user_id: int,
    data: UserUpdate,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(_admin_required),
):
    """Update user fields (role, username, email, active status)."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(404, "User not found")

    if data.username is not None:
        user.username = data.username
    if data.email is not None:
        user.email = data.email
    if data.role is not None:
        if data.role not in ("user", "admin", "reviewer"):
            raise HTTPException(400, "Role must be 'user', 'admin', or 'reviewer'")
        # Guard: cannot demote yourself
        if user.id == admin.id and data.role != "admin":
            raise HTTPException(403, "Cannot demote yourself from admin")
        user.role = data.role
    if data.is_active is not None:
        # Guard: cannot deactivate yourself
        if user.id == admin.id and not data.is_active:
            raise HTTPException(403, "Cannot deactivate yourself")
        user.is_active = data.is_active

    await db.commit()
    await db.refresh(user)
    return {
        "message": f"User {user_id} updated",
        "user": {
            "id": user.id,
            "email": user.email,
            "username": user.username,
            "role": user.role,
            "is_active": getattr(user, "is_active", True),
        },
    }


@router.post("/api/{user_id}/reset-password")
async def admin_reset_password(
    user_id: int,
    data: PasswordReset,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(_admin_required),
):
    """Admin-initiated password reset for a user."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(404, "User not found")

    if len(data.new_password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")

    user.password_hash = hash_password(data.new_password)
    await db.commit()
    return {"message": f"Password reset for user {user_id} ({user.email})"}


@router.post("/api/delete")
async def delete_users(
    data: BulkDelete,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(_admin_required),
):
    """Soft-delete user(s) by setting is_active = False."""
    # Guard: cannot delete yourself
    if admin.id in data.user_ids:
        raise HTTPException(403, "Cannot delete yourself")

    # Guard: cannot delete superuser (id=1)
    if 1 in data.user_ids:
        raise HTTPException(403, "Cannot delete the superuser account")

    count = 0
    for uid in data.user_ids:
        result = await db.execute(select(User).where(User.id == uid))
        user = result.scalar_one_or_none()
        if user:
            user.is_active = False
            count += 1

    await db.commit()
    return {
        "message": f"Soft-deleted {count} user(s)",
        "deleted_ids": data.user_ids,
    }


@router.post("/api/{user_id}/hard-delete")
async def hard_delete_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(_admin_required),
):
    """Permanently delete a user from the database."""
    if user_id == admin.id:
        raise HTTPException(403, "Cannot delete yourself")
    if user_id == 1:
        raise HTTPException(403, "Cannot delete the superuser account")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(404, "User not found")

    await db.delete(user)
    await db.commit()
    return {"message": f"Permanently deleted user {user_id}"}
