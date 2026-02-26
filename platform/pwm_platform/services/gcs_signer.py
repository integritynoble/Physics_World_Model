"""GCS access utility — authenticated proxy for private bucket objects.

Since the GCP project has IAM signing APIs disabled, we serve GCS content
via an authenticated streaming proxy instead of signed URLs.

Provides:
    fetch_gcs_blob(object_path) — download raw bytes from GCS
    fetch_gcs_json(object_path) — download and parse JSON from GCS
    get_blob_metadata(object_path) — get content type and size
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

_client = None
_client_lock = threading.Lock()
_bucket_name: str | None = None


def _get_client():
    """Lazy-initialise google.cloud.storage.Client (thread-safe singleton)."""
    global _client, _bucket_name
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        try:
            from google.cloud import storage

            from pwm_platform.config import settings

            _bucket_name = settings.GCS_BUCKET
            _client = storage.Client()
            logger.info("GCS client initialised (bucket=%s)", _bucket_name)
        except Exception:
            logger.warning("GCS client unavailable", exc_info=True)
            _client = None
    return _client


def fetch_gcs_blob(object_path: str) -> bytes | None:
    """Download raw bytes from GCS using authenticated access.

    Returns bytes or None on failure.
    """
    client = _get_client()
    if client is None or _bucket_name is None:
        return None
    try:
        bucket = client.bucket(_bucket_name)
        blob = bucket.blob(object_path)
        return blob.download_as_bytes(timeout=30)
    except Exception:
        logger.warning("Failed to fetch gs://%s/%s", _bucket_name, object_path, exc_info=True)
        return None


def fetch_gcs_json(object_path: str) -> Any | None:
    """Download and parse a JSON object from GCS using authenticated access.

    Returns parsed JSON or None on failure.
    """
    data = fetch_gcs_blob(object_path)
    if data is None:
        return None
    try:
        return json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.warning("Failed to parse JSON from gs://%s/%s", _bucket_name, object_path)
        return None


def get_blob_metadata(object_path: str) -> dict | None:
    """Get content type and size for a GCS object.

    Returns {"content_type": str, "size": int} or None on failure.
    """
    client = _get_client()
    if client is None or _bucket_name is None:
        return None
    try:
        bucket = client.bucket(_bucket_name)
        blob = bucket.blob(object_path)
        blob.reload()
        return {
            "content_type": blob.content_type or "application/octet-stream",
            "size": blob.size,
        }
    except Exception:
        return None
