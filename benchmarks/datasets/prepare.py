"""CLI entry point to prepare benchmark datasets.

Usage::

    # Prepare all datasets
    python -m benchmarks.datasets.prepare --all

    # Prepare for specific category
    python -m benchmarks.datasets.prepare --category compressive_imaging

    # Prepare for specific modality
    python -m benchmarks.datasets.prepare --modality cassi

    # Dry-run: show what would be downloaded
    python -m benchmarks.datasets.prepare --all --dry-run

    # Upload large datasets to GCS after preparation
    python -m benchmarks.datasets.prepare --all --upload-gcs

    # Force re-download even if cached
    python -m benchmarks.datasets.prepare --modality ct --force
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from benchmarks.datasets.registry import (
    DATASET_REGISTRY,
    DatasetEntry,
    get_all_gcs_entries,
    get_all_local_entries,
    get_datasets_for_category,
    get_datasets_for_modality,
)
from benchmarks.datasets.downloaders import CACHE_ROOT, acquire_dataset
from benchmarks.datasets.gcs_store import GCSDatasetStore

logger = logging.getLogger(__name__)

MANIFEST_PATH = Path(__file__).parent / "manifest.yaml"


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

def load_manifest() -> Dict:
    """Load the manifest YAML."""
    if not MANIFEST_PATH.exists():
        return {"datasets": {}}
    with open(MANIFEST_PATH) as f:
        data = yaml.safe_load(f) or {}
    if "datasets" not in data:
        data["datasets"] = {}
    return data


def save_manifest(manifest: Dict) -> None:
    """Write the manifest YAML."""
    with open(MANIFEST_PATH, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False, width=120)
    logger.info("Updated manifest: %s", MANIFEST_PATH)


def update_manifest_entry(
    manifest: Dict,
    entry: DatasetEntry,
    local_path: Optional[Path],
    gcs_path: Optional[str],
    status: str,
    x_shape: Optional[List[int]] = None,
) -> None:
    """Add or update a single entry in the manifest dict."""
    manifest["datasets"][entry.id] = {
        "source_type": entry.source_type,
        "source_url": entry.url,
        "citation": entry.citation,
        "license": entry.license,
        "local_path": str(local_path) if local_path else None,
        "gcs_path": gcs_path,
        "x_shape": x_shape or entry.x_shape,
        "acquired_date": datetime.date.today().isoformat(),
        "applies_to": entry.applies_to,
        "status": status,
    }


# ---------------------------------------------------------------------------
# Preparation logic
# ---------------------------------------------------------------------------

def prepare_entries(
    entries: List[DatasetEntry],
    dry_run: bool = False,
    upload_gcs: bool = False,
    force: bool = False,
) -> Dict[str, str]:
    """Download, convert, and optionally upload a list of dataset entries.

    Returns a dict mapping ``entry.id`` → status string.
    """
    manifest = load_manifest()
    gcs = GCSDatasetStore() if upload_gcs else None
    results: Dict[str, str] = {}

    for entry in entries:
        print(f"\n{'=' * 60}")
        print(f"  {entry.id}: {entry.name}")
        print(f"  Type: {entry.source_type}  Format: {entry.format}  "
              f"Size: ~{entry.size_mb:.0f} MB  Storage: {entry.storage}")
        print(f"  URL: {entry.url[:80]}{'...' if len(entry.url) > 80 else ''}")
        print(f"  Applies to: {', '.join(entry.applies_to[:8])}"
              f"{'...' if len(entry.applies_to) > 8 else ''}")
        print(f"{'=' * 60}")

        if dry_run:
            cached = (CACHE_ROOT / f"{entry.id}.npy").exists()
            status = "cached" if cached else "pending"
            print(f"  [DRY-RUN] Status: {status}")
            results[entry.id] = status
            update_manifest_entry(manifest, entry, None, None, status)
            continue

        # Skip GCS-only datasets when not uploading and not cached locally
        if entry.storage == "gcs" and not upload_gcs:
            cached_path = CACHE_ROOT / f"{entry.id}.npy"
            if cached_path.exists() and not force:
                print(f"  [CACHED] Using local cache")
                results[entry.id] = "cached"
                update_manifest_entry(
                    manifest, entry, cached_path, None, "acquired",
                )
                continue
            # Still try to download — might work for smaller subsets
            pass

        try:
            npy_path, arr = acquire_dataset(entry, force=force)
            actual_shape = list(arr.shape)
            print(f"  [OK] Acquired: {npy_path.name}  shape={actual_shape}")

            gcs_path = None
            if upload_gcs and entry.storage == "gcs" and gcs is not None:
                category = _infer_category(entry)
                gcs_key = gcs.gcs_key_for(category, entry.id, f"{entry.id}.npy")
                gcs_path = gcs.upload(npy_path, gcs_key)
                if gcs_path:
                    print(f"  [GCS] Uploaded to {gcs_path}")
                else:
                    print(f"  [GCS] Upload failed (continuing)")

            update_manifest_entry(
                manifest, entry, npy_path, gcs_path, "acquired",
                x_shape=actual_shape,
            )
            results[entry.id] = "acquired"

        except Exception as e:
            logger.error("Failed to acquire %s: %s", entry.id, e)
            print(f"  [FAIL] {e}")
            update_manifest_entry(manifest, entry, None, None, "failed")
            results[entry.id] = f"failed: {e}"

    save_manifest(manifest)
    return results


def _infer_category(entry: DatasetEntry) -> str:
    """Infer a category slug from the entry's applies_to list."""
    modality_to_category = {
        "cassi": "compressive_imaging", "cacti": "compressive_imaging",
        "spc": "compressive_imaging", "cup": "compressive_imaging",
        "ct": "medical_ct", "cbct": "medical_ct", "helical_ct": "medical_ct",
        "mri": "medical_mri", "fmri": "medical_mri", "dti": "medical_mri",
        "pet": "nuclear_emission", "spect": "nuclear_emission",
        "cryo_em": "electron_microscopy", "tem": "electron_microscopy",
        "sar": "remote_sensing", "insar": "remote_sensing",
        "afm": "scanning_probe", "stm": "scanning_probe",
        "oct": "optical", "holography": "optical", "lidar": "optical",
        "widefield": "microscopy", "sim": "microscopy", "palm": "microscopy",
    }
    for m in entry.applies_to:
        if m in modality_to_category:
            return modality_to_category[m]
    return "other"


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: Dict[str, str]) -> None:
    """Print a summary table of preparation results."""
    print(f"\n{'=' * 60}")
    print("  PREPARATION SUMMARY")
    print(f"{'=' * 60}")

    acquired = sum(1 for v in results.values() if v == "acquired")
    cached = sum(1 for v in results.values() if v == "cached")
    failed = sum(1 for v in results.values() if v.startswith("failed"))
    pending = sum(1 for v in results.values() if v == "pending")

    print(f"  Total entries:  {len(results)}")
    print(f"  Acquired:       {acquired}")
    print(f"  Cached:         {cached}")
    print(f"  Pending:        {pending}")
    print(f"  Failed:         {failed}")
    print()

    if failed:
        print("  Failed datasets:")
        for k, v in results.items():
            if v.startswith("failed"):
                print(f"    - {k}: {v}")
        print()

    # Coverage: how many modalities are served
    served = set()
    for entry_id, status in results.items():
        if status in ("acquired", "cached"):
            entry = DATASET_REGISTRY.get(entry_id)
            if entry:
                served.update(entry.applies_to)
    print(f"  Modalities covered by real/cached data: {len(served)}")
    if served:
        for m in sorted(served)[:20]:
            print(f"    - {m}")
        if len(served) > 20:
            print(f"    ... and {len(served) - 20} more")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare benchmark datasets: download, convert, cache.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Prepare all datasets")
    group.add_argument("--category", type=str, help="Prepare datasets for a category")
    group.add_argument("--modality", type=str, help="Prepare datasets for a modality")
    group.add_argument("--dataset", type=str, help="Prepare a specific dataset by ID")
    group.add_argument("--list", action="store_true", help="List all registered datasets")

    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--upload-gcs", action="store_true", help="Upload large datasets to GCS")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    parser.add_argument("--local-only", action="store_true",
                        help="Only prepare datasets marked for local storage")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.list:
        _print_registry()
        return

    # Select entries
    if args.all:
        entries = list(DATASET_REGISTRY.values())
    elif args.category:
        entries = get_datasets_for_category(args.category)
    elif args.modality:
        entries = get_datasets_for_modality(args.modality)
    elif args.dataset:
        entry = DATASET_REGISTRY.get(args.dataset)
        if not entry:
            print(f"Unknown dataset: {args.dataset}")
            print(f"Available: {', '.join(DATASET_REGISTRY.keys())}")
            sys.exit(1)
        entries = [entry]
    else:
        entries = []

    if args.local_only:
        entries = [e for e in entries if e.storage == "local"]

    if not entries:
        print("No matching datasets found.")
        sys.exit(1)

    print(f"Preparing {len(entries)} dataset(s)...")
    results = prepare_entries(
        entries,
        dry_run=args.dry_run,
        upload_gcs=args.upload_gcs,
        force=args.force,
    )
    print_summary(results)


def _print_registry():
    """Print the full dataset registry as a table."""
    print(f"\n{'ID':<30} {'Type':<12} {'Format':<8} {'Size MB':>8} {'Storage':<8} {'Modalities'}")
    print("-" * 100)
    for entry in DATASET_REGISTRY.values():
        mods = ", ".join(entry.applies_to[:5])
        if len(entry.applies_to) > 5:
            mods += f" +{len(entry.applies_to) - 5}"
        print(f"{entry.id:<30} {entry.source_type:<12} {entry.format:<8} "
              f"{entry.size_mb:>8.0f} {entry.storage:<8} {mods}")
    print(f"\nTotal: {len(DATASET_REGISTRY)} datasets")


if __name__ == "__main__":
    main()
