#!/usr/bin/env python3
"""Ensure platformaigpt@gmail.com has admin role.

Creates the user if it doesn't exist, or upgrades an existing user to admin.

Usage:
    python scripts/ensure_admin.py
"""

import asyncio
import sys
from pathlib import Path

# Ensure the platform package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from pwm_platform.db.database import async_session_factory, engine, init_db
from pwm_platform.db.models import User

ADMIN_EMAIL = "platformaigpt@gmail.com"
ADMIN_USERNAME = "platformai-admin"


async def main() -> None:
    await init_db()

    async with async_session_factory() as session:
        result = await session.execute(
            select(User).where(User.email == ADMIN_EMAIL)
        )
        user = result.scalar_one_or_none()

        if user is None:
            user = User(
                email=ADMIN_EMAIL,
                username=ADMIN_USERNAME,
                role="admin",
                is_active=True,
            )
            session.add(user)
            await session.commit()
            print(f"Created admin user: {ADMIN_EMAIL} (id={user.id})")
        elif user.role != "admin":
            user.role = "admin"
            await session.commit()
            print(f"Upgraded {ADMIN_EMAIL} to admin (id={user.id})")
        else:
            print(f"Already admin: {ADMIN_EMAIL} (id={user.id})")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
