"""
Initialize the database — create all tables.
Run: python scripts/init_db.py
"""

import asyncio
import sys
import os

# Ensure platform package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    from pwm_platform.db.database import init_db
    print("Creating database tables...")
    await init_db()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
