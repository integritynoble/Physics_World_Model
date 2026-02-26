"""GCS proxy — serves private GCS objects via authenticated streaming.

Since the GCP project has IAM signing APIs disabled, this endpoint
downloads from GCS server-side and streams to the client, acting as
an authenticated proxy with caching headers.
"""

from __future__ import annotations

import logging
import mimetypes

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from pwm_platform.services.gcs_signer import fetch_gcs_blob

logger = logging.getLogger(__name__)

router = APIRouter(tags=["GCS"])

_ALLOWED_PREFIXES = (
    "benchmark_gallery/",
    "benchmark-data/",
)


@router.get("/gcs/{path:path}")
async def gcs_proxy(path: str):
    """Serve a GCS object via authenticated proxy."""
    if not any(path.startswith(prefix) for prefix in _ALLOWED_PREFIXES):
        raise HTTPException(status_code=403, detail="Path not allowed")

    data = fetch_gcs_blob(path)
    if data is None:
        raise HTTPException(status_code=404, detail="Object not found")

    # Guess content type from extension
    content_type, _ = mimetypes.guess_type(path)
    if content_type is None:
        content_type = "application/octet-stream"

    return Response(
        content=data,
        media_type=content_type,
        headers={"Cache-Control": "public, max-age=3600"},
    )
