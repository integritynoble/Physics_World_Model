"""End-to-end test of login + PWM token system using SQLite + TestClient."""
import os, sys, asyncio, secrets

os.environ["SECRET_KEY"] = secrets.token_hex(32)
os.environ["CSRF_SECRET"] = secrets.token_hex(32)
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["GOOGLE_CLIENT_ID"] = ""
os.environ["SSO_VALIDATE_URL"] = ""
os.environ["SSO_REDIRECT_URL"] = ""
os.environ["GCS_BUCKET"] = "test-bucket"

# Patch JSONB → JSON for SQLite BEFORE any model imports
import sqlalchemy.dialects.postgresql as pg
from sqlalchemy import JSON
pg.JSONB = JSON  # type: ignore

# Monkey-patch create_async_engine to drop pg-only kwargs for sqlite
from sqlalchemy.ext.asyncio import create_async_engine as _orig_create
import sqlalchemy.ext.asyncio
def _patched_create(*args, **kw):
    if args and "sqlite" in args[0]:
        kw.pop("pool_size", None); kw.pop("max_overflow", None)
    return _orig_create(*args, **kw)
sqlalchemy.ext.asyncio.create_async_engine = _patched_create

sys.path.insert(0, ".")

# Patch init_db to skip postgres-only ALTERs
from pwm_platform.db import database as db_mod
async def _init_db_patched():
    from pwm_platform.db.database import Base
    from pwm_platform.db import models  # register models
    async with db_mod.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
db_mod.init_db = _init_db_patched

from fastapi.testclient import TestClient
from pwm_platform.main import app
asyncio.get_event_loop().run_until_complete(_init_db_patched())

client = TestClient(app)

# ── Tests ─────────────────────────────────────────────────────────────────
results = []
def check(name, cond, info=""):
    results.append((name, cond, info))
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {name}" + (f"  [{info}]" if not cond and info else ""))

# Test 1: signup
r = client.post("/api/v1/auth/signup-form", data={
    "email": "alice@test.com", "username": "alice", "password": "password123",
}, follow_redirects=False)
check("signup-form returns 303", r.status_code == 303, f"status={r.status_code} body={r.text[:200]}")
check("signup sets cookie", "access_token" in r.cookies)

# Test 2: login with same creds
r = client.post("/api/v1/auth/login-form", data={
    "email": "alice@test.com", "password": "password123",
}, follow_redirects=False)
check("login-form returns 303", r.status_code == 303, f"status={r.status_code}")
jwt = r.cookies.get("access_token")
check("login sets cookie", jwt is not None)

# Test 3: bad password
r = client.post("/api/v1/auth/login-form", data={
    "email": "alice@test.com", "password": "WRONG",
}, follow_redirects=False)
check("bad password → 401", r.status_code == 401, f"status={r.status_code}")

# Test 4: open redirect protection
r = client.post("/api/v1/auth/login-form", data={
    "email": "alice@test.com", "password": "password123", "next": "//evil.com",
}, follow_redirects=False)
loc = r.headers.get("location", "")
check("blocks // open-redirect", loc == "/benchmark", f"loc={loc}")

# Test 5: /auth/me with JWT cookie
c2 = TestClient(app); c2.cookies.set("access_token", jwt)
r = c2.get("/api/v1/auth/me")
check("/auth/me with cookie", r.status_code == 200)

# Test 6: API key generate
r = c2.post("/api/v1/auth/api-key/generate")
check("api-key/generate 200", r.status_code == 200)
api_key = r.json().get("api_key", "")
check("api-key has pwm_ prefix", api_key.startswith("pwm_"))

# Test 7: auth via Bearer api_key
c3 = TestClient(app)
r = c3.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {api_key}"})
check("auth via API key Bearer", r.status_code == 200)

# Test 8: GET masked api-key
r = c2.get("/api/v1/auth/api-key")
d = r.json()
check("api-key GET returns masked", d.get("has_key") and "..." in (d.get("masked_key") or ""))

# Test 9: PWM balance starts at 0
r = c2.get("/api/v1/pwm-token/balance")
check("PWM balance 200", r.status_code == 200)
check("initial balance is 0.0", r.json().get("balance") == 0.0)

# Test 10: reward schedule public
r = client.get("/api/v1/pwm-token/reward-schedule")
check("reward-schedule public", r.status_code == 200)
check("principle reward 1000", r.json().get("rewards", {}).get("principle") == 1000.0)

# Test 11: non-admin cannot award
r = c2.post("/api/v1/pwm-token/award", json={"user_id": 1, "amount": 100, "description": "x"})
check("non-admin → 403", r.status_code == 403, f"status={r.status_code}")

# Test 12: Create admin user, login, award
from pwm_platform.auth.passwords import hash_password
from pwm_platform.db.models import User
from sqlalchemy import select
async def make_admin():
    async with db_mod.async_session_factory() as s:
        u = User(email="admin@t.com", username="admin", role="admin",
                 password_hash=hash_password("adminpass1"))
        s.add(u); await s.commit()
asyncio.get_event_loop().run_until_complete(make_admin())

r = client.post("/api/v1/auth/login-form", data={
    "email": "admin@t.com", "password": "adminpass1",
}, follow_redirects=False)
admin_jwt = r.cookies.get("access_token")
ac = TestClient(app); ac.cookies.set("access_token", admin_jwt)

r = ac.post("/api/v1/pwm-token/award", json={
    "user_id": 1, "amount": 250.0, "description": "Bug bounty"
})
check("admin award 200", r.status_code == 200, f"status={r.status_code} body={r.text[:200]}")
check("award balance_after=250", r.json().get("balance_after") == 250.0)

# Test 13: alice sees balance
r = c2.get("/api/v1/pwm-token/balance")
check("alice balance now 250", r.json().get("balance") == 250.0)
check("alice lifetime=250", r.json().get("lifetime_earned") == 250.0)

# Test 14: transactions
r = c2.get("/api/v1/pwm-token/transactions")
check("alice has 1 txn", len(r.json().get("transactions", [])) == 1)

# Test 15: Create submission + promote-to-mainnet
from pwm_platform.db.models import ChallengeSubmission
async def create_sub():
    async with db_mod.async_session_factory() as s:
        sub = ChallengeSubmission(
            submission_id="sub_001",
            submission_type="principle",
            variant_key="ct_radon",
            tier_name="public",
            submitted_by=1,
            method_name="Test",
            file_path="/tmp/x",
            original_filename="x.json",
            trust_tier="reproduced",
        )
        s.add(sub); await s.commit()
asyncio.get_event_loop().run_until_complete(create_sub())

r = ac.post("/api/v1/pwm-token/promote-to-mainnet",
            json={"submission_id": "sub_001", "comment": "approved"})
check("promote-to-mainnet 200", r.status_code == 200, f"status={r.status_code} body={r.text[:200]}")
d = r.json()
check("reward = 1000 for principle", d.get("reward_amount") == 1000.0)
check("deploy_status=mainnet", d.get("deploy_status") == "mainnet")

# Test 16: idempotency
ac.post("/api/v1/pwm-token/promote-to-mainnet",
        json={"submission_id": "sub_001", "comment": "re-promote"})
r = c2.get("/api/v1/pwm-token/balance")
lt = r.json().get("lifetime_earned")
check("no double-award (lifetime=1250)", lt == 1250.0, f"lifetime={lt}")

# Test 17: set wallet
r = c2.post("/api/v1/pwm-token/wallet", json={"address": "0xabc"})
check("set wallet 200", r.status_code == 200)
check("wallet stored", r.json().get("on_chain_address") == "0xabc")

# Test 18: leaderboard
r = client.get("/api/v1/pwm-token/leaderboard")
lb = r.json().get("leaderboard", [])
check("leaderboard includes alice", any(e.get("username") == "alice" for e in lb))

# Test 19: bad JWT → 401
cb = TestClient(app); cb.cookies.set("access_token", "garbage")
r = cb.get("/api/v1/auth/me")
check("garbage token → 401", r.status_code == 401)

# Test 20: revoke API key
r = c2.delete("/api/v1/auth/api-key")
check("revoke API key", r.status_code == 200)
# After revoke, the old key should not work
r = c3.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {api_key}"})
check("revoked API key → 401", r.status_code == 401, f"status={r.status_code}")

# Test 21: trust_promotion path also awards (via existing /trust/promote endpoint)
async def create_sub2():
    async with db_mod.async_session_factory() as s:
        sub = ChallengeSubmission(
            submission_id="sub_002",
            submission_type="solution",
            variant_key="ct_radon",
            tier_name="public",
            submitted_by=1,
            method_name="Test2",
            file_path="/tmp/y",
            original_filename="y.json",
            trust_tier="reproduced",
        )
        s.add(sub); await s.commit()
asyncio.get_event_loop().run_until_complete(create_sub2())

r = ac.post("/trust/promote/sub_002",
            json={"target_tier": "certified", "comment": "ok"})
check("/trust/promote 200", r.status_code == 200, f"status={r.status_code} body={r.text[:200]}")
check("/trust/promote includes pwm_token_reward",
      "pwm_token_reward" in r.json(),
      f"keys={list(r.json().keys())}")
check("/trust/promote awards 100 PWM for solution",
      r.json().get("pwm_token_reward", {}).get("amount") == 100.0)

# ── Modification submission reward ───────────────────────────────────────

# Test 22-25: award-modification via ChallengeSubmission (sub_002 belongs to user 1 = alice)
# sub_002 was created earlier in the trust_promote test (solution, submitted_by=1)
r = ac.post("/api/v1/pwm-token/award-modification", json={
    "submission_id": "sub_002",
    "amount": 2.5,
    "comment": "Good improvement",
})
check("award-modification 200", r.status_code == 200, f"status={r.status_code} body={r.text[:300]}")
body_mod = r.json()
check("award-modification amount=2.5", body_mod.get("amount_awarded") == 2.5,
      f"amount={body_mod.get('amount_awarded')}")

# Idempotency — same submission_id must return the same transaction
r2 = ac.post("/api/v1/pwm-token/award-modification", json={
    "submission_id": "sub_002",
    "amount": 2.5,
    "comment": "re-award attempt",
})
check("award-modification idempotency: same txn",
      r2.json().get("transaction_id") == body_mod.get("transaction_id"),
      f"first={body_mod.get('transaction_id')} second={r2.json().get('transaction_id')}")

# Out-of-range amount → 422 (Pydantic rejects it before hitting the service)
r3 = ac.post("/api/v1/pwm-token/award-modification", json={
    "submission_id": "sub_002",
    "amount": 99.0,
    "comment": "too much",
})
check("award-modification out-of-range → 422", r3.status_code == 422,
      f"status={r3.status_code}")

# Non-existent submission → 404
r4 = ac.post("/api/v1/pwm-token/award-modification", json={
    "submission_id": "nonexistent_sub_xyz",
    "amount": 1.0,
})
check("award-modification unknown submission → 404", r4.status_code == 404,
      f"status={r4.status_code}")

# ── Paper review access + spend ──────────────────────────────────────────

# Test 22: paper-access when balance is sufficient for deep review
# alice now has 1250 PWM (balance from earlier tests), well above the 10 PWM deep cost
r = c2.get("/api/v1/pwm-token/paper-access")
check("paper-access 200", r.status_code == 200, f"status={r.status_code}")
check("paper-access can_afford_deep=True", r.json().get("can_afford_deep") is True,
      f"body={r.json()}")
check("paper-access review_level=deep", r.json().get("review_level") == "deep")

# Test 25: spend tokens for deep paper review
r = c2.post("/api/v1/pwm-token/spend", json={
    "amount": 10.0,
    "purpose": "paper_review_deep",
    "provider_wallet": "0xa53F7e7Bc6B0Cc182d048217646082DDB2DacfE3",
    "idempotency_key": "paper-test-001",
})
check("spend 200", r.status_code == 200, f"status={r.status_code} body={r.text[:200]}")
body = r.json()
check("spend balance_after decremented", body.get("balance_after") == body.get("balance_before") - 10.0,
      f"before={body.get('balance_before')} after={body.get('balance_after')}")

# Test 26: spend idempotency — same idempotency_key must not double-charge
r2 = c2.post("/api/v1/pwm-token/spend", json={
    "amount": 10.0,
    "purpose": "paper_review_deep",
    "provider_wallet": "0xa53F7e7Bc6B0Cc182d048217646082DDB2DacfE3",
    "idempotency_key": "paper-test-001",
})
check("spend idempotency: same key → same txn",
      r2.json().get("transaction_id") == body.get("transaction_id"),
      f"first={body.get('transaction_id')} second={r2.json().get('transaction_id')}")

# Test 27: spend with insufficient balance → 402
# Create a fresh user with 0 balance, extract JWT, and attempt spend
client.post("/api/v1/auth/signup-form",
            data={"username": "broke_user", "email": "broke@example.com",
                  "password": "pw123456", "next": "/"})
r_login_broke = client.post("/api/v1/auth/login-form",
                             data={"email": "broke@example.com", "password": "pw123456",
                                   "next": "/"}, follow_redirects=False)
broke_jwt = r_login_broke.cookies.get("access_token")
cb2 = TestClient(app)
cb2.cookies.set("access_token", broke_jwt)
r_broke = cb2.post("/api/v1/pwm-token/spend", json={
    "amount": 10.0,
    "purpose": "paper_review_deep",
    "provider_wallet": "0xa53F7e7Bc6B0Cc182d048217646082DDB2DacfE3",
})
check("spend with 0 balance → 402", r_broke.status_code == 402,
      f"status={r_broke.status_code} body={r_broke.text[:200]}")

# Summary
print()
passed = sum(1 for _, c, _ in results if c)
print(f"{'='*60}\n{passed}/{len(results)} tests passed\n{'='*60}")
sys.exit(0 if passed == len(results) else 1)
