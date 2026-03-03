"""Helper to download challenge datasets from GCS for benchmark scripts.

Usage in benchmark scripts:
    from gcs_dataset_helper import ensure_challenge_dataset
    h5_path = ensure_challenge_dataset("ct", "public")
    # Returns Path to local .h5 file (downloaded from GCS if not cached)
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

GCS_BUCKET = "pwm-benchmark-datasets"
GCS_PREFIX = "challenge-data/v1.0"
CACHE_DIR = Path("/tmp/pwm_challenge_cache")


def ensure_challenge_dataset(
    variant: str,
    tier: str,
    cache_dir: Path | None = None,
) -> Path:
    """Download a challenge HDF5 file from GCS if not already cached locally.

    Parameters
    ----------
    variant : str
        Challenge variant key (e.g. "ct", "sd_cassi", "cacti", "spc_block").
    tier : str
        Tier name: "public", "dev", or "hidden".
    cache_dir : Path or None
        Local cache directory. Defaults to /tmp/pwm_challenge_cache/.

    Returns
    -------
    Path
        Path to local .h5 file (downloaded from GCS or from cache).

    Raises
    ------
    RuntimeError
        If the file cannot be downloaded and is not cached locally.
    """
    cache = cache_dir or CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)

    filename = f"{variant}_challenge_{tier}.h5"
    local_path = cache / filename

    # Use cache if available
    if local_path.exists() and local_path.stat().st_size > 0:
        logger.info("Using cached dataset: %s", local_path)
        return local_path

    # Download from GCS
    gcs_key = f"{GCS_PREFIX}/{filename}"
    logger.info("Downloading gs://%s/%s -> %s", GCS_BUCKET, gcs_key, local_path)

    try:
        from google.cloud import storage as gcs_storage

        client = gcs_storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(gcs_key)

        if not blob.exists():
            raise RuntimeError(
                f"GCS object not found: gs://{GCS_BUCKET}/{gcs_key}"
            )

        blob.download_to_filename(str(local_path))
        logger.info("Downloaded %s (%.1f MB)", filename, local_path.stat().st_size / 1e6)
        return local_path

    except ImportError:
        raise RuntimeError(
            "google-cloud-storage not installed. "
            "Run: pip install google-cloud-storage"
        )
    except Exception as e:
        # Clean up partial download
        if local_path.exists() and local_path.stat().st_size == 0:
            local_path.unlink()
        raise RuntimeError(f"Failed to download {filename} from GCS: {e}")
