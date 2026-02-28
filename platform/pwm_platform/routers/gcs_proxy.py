"""GCS proxy — serves private GCS objects via authenticated streaming.

Since the GCP project has IAM signing APIs disabled, this endpoint
downloads from GCS server-side and streams to the client, acting as
an authenticated proxy with caching headers.

Falls back to local static files when GCS is unavailable.

Security: Hidden-tier challenge files (containing ground truth) are
blocked from public download. Only public and dev tier files are served.
"""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path, PurePosixPath

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from pwm_platform.services.gcs_signer import fetch_gcs_blob

logger = logging.getLogger(__name__)

router = APIRouter(tags=["GCS"])

_ALLOWED_PREFIXES = (
    "benchmark_gallery/",
    "benchmark-data/",
    "challenge-data/",
)

# Filename patterns that are BLOCKED from public download (contain ground truth)
_BLOCKED_PATTERNS = (
    "_challenge_hidden",   # hidden tier files contain x_true + true_spec
)

# Local static fallback directory
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static" / "benchmark-data"

# Extensions that trigger Content-Disposition: attachment (file download)
_DOWNLOAD_EXTENSIONS = {".h5", ".hdf5", ".npz", ".npy", ".zip", ".tar", ".gz"}


@router.get("/gcs/{path:path}")
async def gcs_proxy(path: str):
    """Serve a GCS object via authenticated proxy, with local fallback."""
    if not any(path.startswith(prefix) for prefix in _ALLOWED_PREFIXES):
        raise HTTPException(status_code=403, detail="Path not allowed")

    # Block access to hidden tier challenge files (contain ground truth)
    filename = PurePosixPath(path).name
    if any(pattern in filename for pattern in _BLOCKED_PATTERNS):
        logger.warning("Blocked access to restricted file: %s", path)
        raise HTTPException(
            status_code=403,
            detail="Hidden tier data is not available for public download",
        )

    suffix = PurePosixPath(path).suffix.lower()
    is_download = suffix in _DOWNLOAD_EXTENSIONS

    # Try local static files first (avoids GCS latency + memory for large files)
    local_path = _STATIC_DIR / path
    if local_path.is_file():
        logger.info("Serving %s from local static", path)
        content_type, _ = mimetypes.guess_type(path)
        headers = {"Cache-Control": "public, max-age=3600"}
        if is_download:
            headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        return FileResponse(
            path=str(local_path),
            media_type=content_type or "application/octet-stream",
            headers=headers,
            filename=filename if is_download else None,
        )

    # Fallback to GCS
    data = fetch_gcs_blob(path)
    if data is None:
        raise HTTPException(status_code=404, detail="Object not found")

    content_type, _ = mimetypes.guess_type(path)
    if content_type is None:
        content_type = "application/octet-stream"

    headers = {"Cache-Control": "public, max-age=3600"}
    if is_download:
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'

    return Response(
        content=data,
        media_type=content_type,
        headers=headers,
    )
