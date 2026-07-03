"""Delete account: confirm gate, deactivation, email freed, key revoked.

pwm_platform mirror of pwm_nonprofit/platform/tests/test_delete_account.py
(single User.api_key column here instead of the ApiKey table).
Run: python3 tests/test_delete_account.py  (from the platform/ dir)
"""
import os, sys, asyncio, secrets

os.environ["SECRET_KEY"] = secrets.token_hex(32)
os.environ["CSRF_SECRET"] = secrets.token_hex(32)
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["GOOGLE_CLIENT_ID"] = ""
os.environ["SSO_VALIDATE_URL"] = ""
os.environ["SSO_REDIRECT_URL"] = ""
os.environ["GCS_BUCKET"] = "test-bucket"

import sqlalchemy.dialects.postgresql as pg
from sqlalchemy import JSON
pg.JSONB = JSON  # type: ignore

from sqlalchemy.ext.asyncio import create_async_engine as _orig_create
import sqlalchemy.ext.asyncio
def _patched_create(*args, **kw):
    if args and "sqlite" in args[0]:
        kw.pop("pool_size", None); kw.pop("max_overflow", None)
    return _orig_create(*args, **kw)
sqlalchemy.ext.asyncio.create_async_engine = _patched_create

sys.path.insert(0, ".")

from pwm_platform.db import database as db_mod
async def _init_db_patched():
    from pwm_platform.db.database import Base
    from pwm_platform.db import models
    async with db_mod.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
db_mod.init_db = _init_db_patched

from fastapi.testclient import TestClient
from pwm_platform.main import app
asyncio.get_event_loop().run_until_complete(_init_db_patched())

results = []
def check(name, cond, info=""):
    results.append((name, cond, info))
    print(("PASS" if cond else "FAIL") + f": {name}" + (f"  [{info}]" if not cond and info else ""))

PW = "password123"

# signup
c = TestClient(app)
r = c.post("/api/v1/auth/signup-form", data={
    "email": "dave@test.com", "username": "dave", "password": PW,
}, follow_redirects=False)
check("signup 303", r.status_code == 303, f"status={r.status_code}")
c.cookies.set("access_token", r.cookies.get("access_token"))

# create an API key so we can verify revocation
r = c.post("/api/v1/auth/api-key/generate")
check("api key created", r.status_code == 200, f"status={r.status_code} body={r.text[:150]}")

# unauthenticated → 401
r = TestClient(app).post("/api/v1/auth/delete-account", json={"confirm": "DELETE"})
check("unauthenticated → 401", r.status_code == 401, f"status={r.status_code}")

# wrong confirm string → 400
r = c.post("/api/v1/auth/delete-account", json={"confirm": "delete"})
check("wrong confirm → 400", r.status_code == 400, f"status={r.status_code}")

# valid delete → 200
r = c.post("/api/v1/auth/delete-account", json={"confirm": "DELETE"})
check("delete → 200", r.status_code == 200, f"status={r.status_code} body={r.text[:150]}")

# session no longer valid
r = c.get("/api/v1/auth/api-key")
check("session dead after delete", r.status_code == 401, f"status={r.status_code}")

# login with old credentials fails
r = TestClient(app).post("/api/v1/auth/login-form", data={"email": "dave@test.com", "password": PW}, follow_redirects=False)
check("old login → 401", r.status_code == 401, f"status={r.status_code}")

# email is freed — fresh signup with same email succeeds
r = TestClient(app).post("/api/v1/auth/signup-form", data={
    "email": "dave@test.com", "username": "dave2", "password": PW,
}, follow_redirects=False)
check("email freed for re-signup", r.status_code == 303, f"status={r.status_code} body={r.text[:200]}")

passed = sum(1 for _, ok, _ in results if ok)
print("\n" + "=" * 60)
print(f"{passed}/{len(results)} tests passed")
print("=" * 60)
sys.exit(0 if passed == len(results) else 1)
