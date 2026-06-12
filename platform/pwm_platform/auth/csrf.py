"""
CSRF protection for the form-based UI (double-submit cookie + origin check).

Strategy (layered, so programmatic clients keep working):

1. **Double-submit token** — `CSRFTokenMiddleware` puts a signed token in an
   HttpOnly `csrf_token` cookie and exposes it as `request.state.csrf_token`
   so templates embed it as a hidden `csrf_token` form field. A POST whose
   form value matches the cookie (and carries a valid HMAC signature) passes.

2. **Origin / Referer check** — browsers always send an `Origin` header on
   cross-site form POSTs, so a submission whose Origin/Referer host differs
   from the request host is rejected. Same-host submissions pass even
   without a token (covers cached pages rendered before this feature).

3. **Header-less clients pass** — requests with no token AND no
   Origin/Referer (test suites, CLI scripts posting to the form endpoints)
   are allowed: they are not browsers, so they cannot be carrying a
   victim's ambient cookies. This keeps `test_agent_mining_e2e.py`,
   `test_cli_auth.py` and similar callers working unchanged.

Tokens are `nonce.hexsig` where hexsig = HMAC-SHA256(CSRF_SECRET, nonce),
so a token can be authenticated without server-side storage.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from urllib.parse import urlsplit

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

from pwm_platform.config import settings

CSRF_COOKIE_NAME = "csrf_token"

_CSRF_COOKIE_KWARGS = {
    "key": CSRF_COOKIE_NAME,
    "httponly": True,   # templates get the value server-side; JS never needs it
    "secure": True,
    "samesite": "lax",
    "max_age": 7 * 24 * 3600,
    "path": "/",
}


def _sign(nonce: str) -> str:
    return hmac.new(
        settings.CSRF_SECRET.encode(), nonce.encode(), hashlib.sha256
    ).hexdigest()


def generate_csrf_token() -> str:
    """Return a fresh signed CSRF token (`nonce.hexsig`)."""
    nonce = secrets.token_urlsafe(24)
    return f"{nonce}.{_sign(nonce)}"


def validate_csrf_token(token: str) -> bool:
    """Check that a token was signed with our CSRF_SECRET."""
    if not token or "." not in token:
        return False
    nonce, sig = token.rsplit(".", 1)
    return hmac.compare_digest(_sign(nonce), sig)


class CSRFTokenMiddleware(BaseHTTPMiddleware):
    """Issue the csrf_token cookie and expose it to templates.

    `request.state` is backed by the ASGI scope, so the value set here is
    visible to route handlers and to Jinja templates rendered with the same
    request (`{{ request.state.csrf_token }}`).
    """

    async def dispatch(self, request: Request, call_next):
        token = request.cookies.get(CSRF_COOKIE_NAME, "")
        fresh = False
        if not validate_csrf_token(token):
            token = generate_csrf_token()
            fresh = True
        request.state.csrf_token = token
        response = await call_next(request)
        if fresh and request.method == "GET":
            response.set_cookie(value=token, **_CSRF_COOKIE_KWARGS)
        return response


def _origin_host(value: str) -> str | None:
    try:
        return urlsplit(value).hostname
    except ValueError:
        return None


def enforce_csrf(request: Request, form_token: str = "") -> None:
    """Raise 403 on a cross-site form submission. See module docstring."""
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME, "")

    # Layer 1: double-submit token match (signed + identical to cookie).
    if (
        form_token
        and cookie_token
        and hmac.compare_digest(form_token, cookie_token)
        and validate_csrf_token(form_token)
    ):
        return

    # Layer 2: Origin/Referer must match the request host when present.
    # A sandboxed-iframe attack sends "Origin: null" → hostname None → 403.
    origin = request.headers.get("origin") or request.headers.get("referer")
    if origin:
        host = _origin_host(origin)
        allowed = {request.url.hostname}
        allowed.update(
            _origin_host(o) for o in settings.CORS_ORIGINS
        )
        if host and host in allowed:
            return
        raise HTTPException(
            status_code=403,
            detail="CSRF validation failed: cross-origin form submission",
        )

    # Layer 3: a token was submitted but didn't validate, and no origin
    # header to vouch for the request — reject.
    if form_token:
        raise HTTPException(status_code=403, detail="CSRF validation failed")

    # No token, no Origin/Referer: non-browser client. Allow.
    return
