"""
Create an admin user.
Run: python scripts/create_admin.py
"""

import asyncio
import getpass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    from sqlalchemy import select
    from pwm_platform.auth.passwords import hash_password
    from pwm_platform.db.database import async_session_factory, init_db
    from pwm_platform.db.models import User

    await init_db()

    email = input("Admin email: ").strip()
    username = input("Admin username: ").strip()
    password = getpass.getpass("Admin password (min 8 chars): ")

    if len(password) < 8:
        print("Password must be at least 8 characters.")
        return

    async with async_session_factory() as db:
        result = await db.execute(select(User).where(User.email == email))
        if result.scalar_one_or_none():
            print(f"User with email {email} already exists.")
            return

        user = User(
            email=email,
            username=username,
            password_hash=hash_password(password),
            role="admin",
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
        print(f"Admin user created: id={user.id}, email={email}, role=admin")


if __name__ == "__main__":
    asyncio.run(main())
