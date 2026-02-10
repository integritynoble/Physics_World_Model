"""pwm_core.io.caching

Content-addressed caching layer for datasets and operator artifacts.

Provides:
- ensure_cached: Copy file to cache (original behavior)
- content_hash: SHA256 of file content
- cache_get: Look up cached artifact by content hash
- cache_put: Store file in cache by content hash
- cache_stats: Cache size and file count
"""

from __future__ import annotations

import hashlib
import os
import shutil
from typing import Any, Dict, Optional


def content_hash(path: str) -> str:
    """SHA256 of file content."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def ensure_cached(path: str, cache_dir: str) -> str:
    """Copy file to cache if not present (original behavior)."""
    os.makedirs(cache_dir, exist_ok=True)
    dst = os.path.join(cache_dir, os.path.basename(path))
    if not os.path.exists(dst):
        shutil.copy2(path, dst)
    return dst


def cache_get(key: str, cache_dir: str) -> Optional[str]:
    """Look up a cached artifact by content hash.

    Parameters
    ----------
    key : str
        Content hash (SHA256) to look up.
    cache_dir : str
        Cache directory.

    Returns
    -------
    str or None
        Path to cached file if found, None otherwise.
    """
    if not os.path.isdir(cache_dir):
        return None

    # Content-addressed: files stored as <hash>/<original_name>
    hash_dir = os.path.join(cache_dir, key[:2], key)
    if os.path.isdir(hash_dir):
        files = os.listdir(hash_dir)
        if files:
            return os.path.join(hash_dir, files[0])

    # Fallback: check flat cache by hash prefix in filename
    for fname in os.listdir(cache_dir):
        if fname.startswith(key[:16]):
            return os.path.join(cache_dir, fname)

    return None


def cache_put(path: str, cache_dir: str) -> str:
    """Store file in cache by content hash.

    Parameters
    ----------
    path : str
        Source file path.
    cache_dir : str
        Cache directory.

    Returns
    -------
    str
        Path to the cached file.
    """
    os.makedirs(cache_dir, exist_ok=True)
    h = content_hash(path)

    # Store in content-addressed structure: <prefix>/<hash>/<filename>
    hash_dir = os.path.join(cache_dir, h[:2], h)
    os.makedirs(hash_dir, exist_ok=True)

    dst = os.path.join(hash_dir, os.path.basename(path))
    if not os.path.exists(dst):
        shutil.copy2(path, dst)

    return dst


def cache_stats(cache_dir: str) -> Dict[str, Any]:
    """Return cache size, num files, total bytes.

    Parameters
    ----------
    cache_dir : str
        Cache directory.

    Returns
    -------
    dict
        Keys: num_files, total_bytes, cache_dir.
    """
    if not os.path.isdir(cache_dir):
        return {"num_files": 0, "total_bytes": 0, "cache_dir": cache_dir}

    num_files = 0
    total_bytes = 0

    for root, _dirs, files in os.walk(cache_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            num_files += 1
            try:
                total_bytes += os.path.getsize(fpath)
            except OSError:
                pass

    return {
        "num_files": num_files,
        "total_bytes": total_bytes,
        "cache_dir": cache_dir,
    }
