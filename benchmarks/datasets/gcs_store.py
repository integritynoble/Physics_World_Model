"""GCS integration for large benchmark datasets (>100 MB).

Uploads converted ``.npy`` files to Google Cloud Storage and downloads
them back with local caching.  Falls back gracefully when GCS
credentials are unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_BUCKET = "pwm-benchmark-datasets"
DEFAULT_CACHE = Path(__file__).parent.parent / "results" / ".data_cache" / "gcs"


class GCSDatasetStore:
    """Upload/download benchmark datasets to/from Google Cloud Storage.

    Parameters
    ----------
    bucket_name : str
        GCS bucket name.
    cache_dir : Path or None
        Local directory to cache downloaded files.  Defaults to
        ``benchmarks/results/.data_cache/gcs/``.
    """

    def __init__(
        self,
        bucket_name: str = DEFAULT_BUCKET,
        cache_dir: Optional[Path] = None,
    ):
        self.bucket_name = bucket_name
        self.cache_dir = cache_dir or DEFAULT_CACHE
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._bucket = None

    # ------------------------------------------------------------------
    # Lazy client initialisation
    # ------------------------------------------------------------------

    def _get_bucket(self):
        """Return the GCS bucket object, or ``None`` if unavailable."""
        if self._bucket is not None:
            return self._bucket
        try:
            from google.cloud import storage as gcs_storage
            self._client = gcs_storage.Client()
            self._bucket = self._client.bucket(self.bucket_name)
            return self._bucket
        except Exception as e:
            logger.warning(
                "GCS unavailable (%s). Large datasets will not be uploaded/downloaded. "
                "Set GOOGLE_APPLICATION_CREDENTIALS or run `gcloud auth application-default login`.",
                e,
            )
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upload(self, local_path: Path, gcs_key: str) -> Optional[str]:
        """Upload *local_path* to ``gs://<bucket>/<gcs_key>``.

        Returns the ``gs://`` URI on success, ``None`` on failure.
        """
        bucket = self._get_bucket()
        if bucket is None:
            return None

        blob = bucket.blob(gcs_key)
        try:
            blob.upload_from_filename(str(local_path))
            uri = f"gs://{self.bucket_name}/{gcs_key}"
            logger.info("Uploaded %s -> %s", local_path.name, uri)
            return uri
        except Exception as e:
            logger.error("Upload failed for %s: %s", gcs_key, e)
            return None

    def download(self, gcs_key: str, local_path: Optional[Path] = None) -> Optional[Path]:
        """Download ``gs://<bucket>/<gcs_key>`` to *local_path*.

        Uses the local cache by default.  Returns the local path on
        success, ``None`` on failure.
        """
        if local_path is None:
            # Derive from gcs_key
            local_path = self.cache_dir / gcs_key.replace("/", "_")
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Check cache
        if local_path.exists() and local_path.stat().st_size > 0:
            logger.info("GCS cache hit: %s", local_path.name)
            return local_path

        bucket = self._get_bucket()
        if bucket is None:
            return None

        blob = bucket.blob(gcs_key)
        try:
            blob.download_to_filename(str(local_path))
            logger.info("Downloaded gs://%s/%s -> %s", self.bucket_name, gcs_key, local_path)
            return local_path
        except Exception as e:
            logger.error("Download failed for %s: %s", gcs_key, e)
            return None

    def exists(self, gcs_key: str) -> bool:
        """Check whether *gcs_key* exists in the bucket."""
        bucket = self._get_bucket()
        if bucket is None:
            return False
        blob = bucket.blob(gcs_key)
        try:
            return blob.exists()
        except Exception:
            return False

    def list_datasets(self, prefix: str = "datasets/") -> List[str]:
        """List all objects under *prefix* in the bucket."""
        bucket = self._get_bucket()
        if bucket is None:
            return []
        try:
            return [blob.name for blob in bucket.list_blobs(prefix=prefix)]
        except Exception as e:
            logger.error("Listing failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def gcs_key_for(self, category: str, dataset_id: str, filename: str) -> str:
        """Build a canonical GCS key.

        Format: ``datasets/<category>/<dataset_id>/<filename>``
        """
        return f"datasets/{category}/{dataset_id}/{filename}"

    @property
    def available(self) -> bool:
        """Return ``True`` if GCS credentials are configured."""
        return self._get_bucket() is not None
