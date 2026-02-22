"""
JWT Token Manager — ported from CompareGPT-AIScientist.

Creates and verifies HS256 JWT access tokens.
Compatible with the CompareGPT token format (user_id in payload, 7-day expiry, jti).
"""

from __future__ import annotations

import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt

logger = logging.getLogger(__name__)

ALGORITHM = "HS256"


class TokenManager:
    """Create and verify JWT access tokens."""

    def __init__(self, secret_key: str, expire_days: int = 7):
        self.secret_key = secret_key
        self.expire_days = expire_days

    def create_access_token(self, user_id: int) -> str:
        """Create a JWT access token for *user_id*.

        Payload mirrors CompareGPT format:
            user_id, exp, iat, jti
        """
        now = datetime.now(timezone.utc)
        payload = {
            "user_id": user_id,
            "exp": now + timedelta(days=self.expire_days),
            "iat": now,
            "jti": secrets.token_urlsafe(16),
        }
        return jwt.encode(payload, self.secret_key, algorithm=ALGORITHM)

    def verify_access_token(self, token: str) -> Optional[int]:
        """Return *user_id* if the token is valid, else ``None``."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[ALGORITHM])
            return payload.get("user_id")
        except jwt.ExpiredSignatureError:
            logger.debug("Token expired")
            return None
        except jwt.InvalidTokenError as exc:
            logger.debug("Invalid token: %s", exc)
            return None


def _build_token_manager() -> TokenManager:
    """Construct the singleton.  Deferred import avoids circular deps."""
    from pwm_platform.config import settings

    return TokenManager(
        secret_key=settings.SECRET_KEY,
        expire_days=settings.ACCESS_TOKEN_EXPIRE_DAYS,
    )


# Lazy singleton — initialised on first access via the module-level helper
_token_manager: Optional[TokenManager] = None


def get_token_manager() -> TokenManager:
    global _token_manager
    if _token_manager is None:
        _token_manager = _build_token_manager()
    return _token_manager
