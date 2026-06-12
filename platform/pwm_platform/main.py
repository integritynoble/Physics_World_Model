"""
PWM Platform — FastAPI application entry point.

Serves:
  - JSON API at /api/v1/...
  - Server-rendered UI pages at / (Jinja2 + HTMX)
  - OpenAPI docs at /docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from pwm_platform.auth.csrf import CSRFTokenMiddleware
from pwm_platform.auth.dependencies import LoginRedirect

from pwm_platform.config import settings
from pwm_platform.db.database import init_db
from pwm_platform.routers import (
    audit_router,
    admin_users_router,
    auth_router,
    billing_router,
    bootstrap_router,
    claim_review_router,
    contributors_router,
    datasets_router,
    gcs_proxy_router,
    instruments_router,
    modalities_router,
    pages_router,
    runs_router,
    spec_chat_router,
    submissions_router,
    system_design_chat_router,
    trust_promotion_router,
    solvers_router,
    pwm_token_router,
)

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ─────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    import threading

    logger.info("PWM Platform starting up (%s)", settings.APP_VERSION)
    await init_db()
    logger.info("Database tables ensured")

    # Pre-generate example .npy datasets in a background thread so the
    # event loop stays responsive while CPU-heavy NumPy work runs.
    def _warmup():
        from pwm_platform.services.example_datasets import warmup_all
        warmup_all()

    threading.Thread(target=_warmup, daemon=True).start()

    yield
    logger.info("PWM Platform shutting down")


# ── App ──────────────────────────────────────────────────────────────────


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CSRF — issues the signed csrf_token cookie and exposes it to templates
# as request.state.csrf_token. Validation happens in the auth form
# endpoints via enforce_csrf(). See pwm_platform/auth/csrf.py.
app.add_middleware(CSRFTokenMiddleware)

# Static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="pwm_platform/static"), name="static")

# ── Register routers ────────────────────────────────────────────────────

app.include_router(audit_router)
app.include_router(admin_users_router)
app.include_router(auth_router)
app.include_router(billing_router)
app.include_router(claim_review_router)
app.include_router(contributors_router)
app.include_router(runs_router)
app.include_router(datasets_router)
app.include_router(modalities_router)
app.include_router(bootstrap_router)
app.include_router(spec_chat_router)
app.include_router(system_design_chat_router)
app.include_router(gcs_proxy_router)
app.include_router(submissions_router)
app.include_router(trust_promotion_router)
app.include_router(instruments_router)
app.include_router(solvers_router)
app.include_router(pwm_token_router)
app.include_router(pages_router)


# ── Global exception handlers ───────────────────────────────────────────


@app.exception_handler(LoginRedirect)
async def login_redirect_handler(request: Request, exc: LoginRedirect):
    """Redirect unauthenticated page requests to /login?next=<path>."""
    return RedirectResponse(f"/login?next={exc.next_url}", status_code=302)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Redirect unauthenticated UI requests to /login?next=; pass through others."""
    if exc.status_code == 401 and "text/html" in request.headers.get("accept", ""):
        from urllib.parse import quote
        next_path = quote(str(request.url.path), safe="/")
        return RedirectResponse(f"/login?next={next_path}", status_code=302)
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc.detail)})


# ── Health check ─────────────────────────────────────────────────────────


@app.get("/api/v1/health")
async def health():
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "service": "PWM Platform",
    }


# ── Entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "pwm_platform.main:app",
        host="0.0.0.0",
        port=settings.APP_PORT,
        reload=settings.DEBUG,
    )
