#!/usr/bin/env python3
"""Download real brain MRI datasets for the PWM MRI benchmark.

Two datasets are supported:

IXI T2w  (dev tier)  — auto-downloaded, no registration required
─────────────────────────────────────────────────────────────────
  578 healthy T2-weighted brain MRI scans from three London sites:
    Hammersmith Hospital  (3 T, Philips)
    Guy's Hospital        (1.5 T, Philips)
    Institute of Psychiatry (1.5 T, GE)
  Licence:    CC BY-SA 3.0  — free to use, cite IXI paper
  Compressed: ~11 GB  (IXI-T2.tar)
  Reference:  brain-development.org/ixi-dataset/

BraTS 2024  (hidden tier) — manual download, free account required
────────────────────────────────────────────────────────────────────
  1,251+ multi-modal brain MRI cases with radiologist-annotated tumours
  (GBM, LGG, meningioma, metastases)
  Modalities: T1w · T1CE · T2w · T2-FLAIR
  Licence:    research use, Synapse data-use agreement
  Download:   https://www.synapse.org/brats2024
  Reference:  Baid et al., 2021, arXiv:2107.02314

Usage
─────
    # Install nibabel first (required for NIfTI loading):
    pip install nibabel

    # Download IXI T2w automatically:
    python download_datasets.py --ixi-dir ~/pwm_data/ixi_t2

    # Or point to already-extracted directory:
    python download_datasets.py --ixi-dir /data/IXI-T2

    # Then build the benchmark:
    IXI_T2_ROOT=~/pwm_data/ixi_t2 BRATS_ROOT=~/pwm_data/brats2024 python build_dataset.py
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

# ── Download URLs ──────────────────────────────────────────────────────────────

IXI_T2_URL  = "https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T2.tar"
IXI_T2_BYTES_APPROX = 11_500_000_000  # ~11.5 GB

# ── Helpers ────────────────────────────────────────────────────────────────────

def _check_nibabel():
    try:
        import nibabel  # noqa: F401
        return True
    except ImportError:
        return False


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    total = total_size if total_size > 0 else IXI_T2_BYTES_APPROX
    frac = min(downloaded / total, 1.0)
    bar = "#" * int(frac * 40)
    mb_done = downloaded / 1e6
    mb_total = total / 1e6
    print(f"\r  [{bar:<40}] {mb_done:6.0f}/{mb_total:.0f} MB", end="", flush=True)


# ── IXI downloader ─────────────────────────────────────────────────────────────

def download_ixi_t2(ixi_dir: str) -> bool:
    """Download and extract IXI-T2.tar to ixi_dir.

    Skips download if .nii.gz files already present.
    Returns True on success.
    """
    ixi_dir = os.path.expanduser(ixi_dir)
    os.makedirs(ixi_dir, exist_ok=True)

    # Check if already extracted
    existing = list(Path(ixi_dir).glob("*T2*.nii.gz"))
    if len(existing) >= 10:
        print(f"  [IXI] Found {len(existing)} T2 NIfTI files in {ixi_dir} — skipping download.")
        return True

    tar_path = os.path.join(ixi_dir, "IXI-T2.tar")

    # Download if not present / incomplete
    if os.path.exists(tar_path):
        on_disk = os.path.getsize(tar_path)
        if on_disk > 0.9 * IXI_T2_BYTES_APPROX:
            print(f"  [IXI] Tar already present ({on_disk/1e9:.1f} GB) — skipping download.")
        else:
            print(f"  [IXI] Existing tar looks incomplete ({on_disk/1e6:.0f} MB) — re-downloading.")
            os.remove(tar_path)

    if not os.path.exists(tar_path):
        print(f"  [IXI] Downloading IXI-T2.tar (~11.5 GB) from brain-development.org...")
        print(f"  [IXI] URL: {IXI_T2_URL}")
        print(f"  [IXI] Destination: {tar_path}")
        print("  [IXI] This may take a while on slow connections.")
        t0 = time.time()
        try:
            urllib.request.urlretrieve(IXI_T2_URL, tar_path, _progress_hook)
            print()  # newline after progress bar
            elapsed = time.time() - t0
            size_gb = os.path.getsize(tar_path) / 1e9
            print(f"  [IXI] Downloaded {size_gb:.1f} GB in {elapsed/60:.1f} min.")
        except Exception as exc:
            print(f"\n  [ERROR] Download failed: {exc}")
            print("  [IXI] Manual download:")
            print(f"    wget -c '{IXI_T2_URL}' -O '{tar_path}'")
            print(f"    tar -xf '{tar_path}' -C '{ixi_dir}'")
            return False

    # Extract
    print(f"  [IXI] Extracting {tar_path} ...")
    try:
        with tarfile.open(tar_path, "r") as tf:
            members = [m for m in tf.getmembers() if m.name.endswith(".nii.gz")]
            print(f"  [IXI] Extracting {len(members)} NIfTI files...")
            for i, member in enumerate(members):
                tf.extract(member, path=ixi_dir)
                if i % 50 == 0:
                    print(f"  [IXI]   {i}/{len(members)} files extracted...", end="\r")
        print(f"  [IXI] Extraction complete: {ixi_dir}")
    except Exception as exc:
        print(f"  [ERROR] Extraction failed: {exc}")
        return False

    # Count extracted files
    n_files = len(list(Path(ixi_dir).rglob("*T2*.nii.gz")))
    print(f"  [IXI] Found {n_files} T2 NIfTI files after extraction.")
    return n_files > 0


# ── BraTS instructions ─────────────────────────────────────────────────────────

def print_brats_instructions(brats_dir: str):
    """Print step-by-step BraTS 2024 download instructions."""
    print()
    print("=" * 70)
    print("BraTS 2024 — Manual Download Required")
    print("=" * 70)
    print()
    print("BraTS requires a free Synapse account and data-use agreement.")
    print()
    print("Steps:")
    print("  1. Create a free account at https://www.synapse.org/")
    print("  2. Visit the BraTS 2024 challenge page:")
    print("       https://www.synapse.org/brats2024")
    print("  3. Accept the data-use agreement.")
    print("  4. Download the BraTS 2024 GLI (glioma) training dataset:")
    print("       synID: syn53708249")
    print("  5. Extract to a directory, e.g.:")
    print(f"       {brats_dir}")
    print()
    print("  Alternatively, BraTS 2020/2021 datasets are on Kaggle:")
    print("    https://www.kaggle.com/datasets/awsaf49/brats2020-training-data")
    print()
    print("  After downloading, set the environment variable:")
    print(f"    export BRATS_ROOT={brats_dir}")
    print()
    print("  Expected directory structure (any BraTS year works):")
    print("    BraTS-GLI-00000-000/")
    print("      BraTS-GLI-00000-000-t2w.nii.gz     ← T2-weighted")
    print("      BraTS-GLI-00000-000-t1w.nii.gz")
    print("      BraTS-GLI-00000-000-t1c.nii.gz")
    print("      BraTS-GLI-00000-000-t2f.nii.gz     ← T2-FLAIR")
    print()
    print("  Or BraTS 2020 style:")
    print("    BraTS20_Training_001/")
    print("      BraTS20_Training_001_t2.nii.gz")
    print()
    print("Note: Without BraTS, the hidden tier falls back to procedural")
    print("      brain phantoms (which still work well for benchmarking).")
    print("=" * 70)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download real brain MRI data for the PWM MRI benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ixi-dir",
        default=os.environ.get("IXI_T2_ROOT", os.path.expanduser("~/pwm_data/ixi_t2")),
        help="Directory to download/extract IXI T2w NIfTI files into. "
             "Defaults to IXI_T2_ROOT env var or ~/pwm_data/ixi_t2.",
    )
    parser.add_argument(
        "--brats-dir",
        default=os.environ.get("BRATS_ROOT", os.path.expanduser("~/pwm_data/brats2024")),
        help="Directory where BraTS data is (or will be) located. "
             "Defaults to BRATS_ROOT env var or ~/pwm_data/brats2024.",
    )
    parser.add_argument(
        "--skip-ixi",
        action="store_true",
        help="Skip IXI download (use if you already have the data).",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("PWM MRI Benchmark — Dataset Downloader")
    print("=" * 70)

    # Check nibabel
    if not _check_nibabel():
        print()
        print("[WARNING] nibabel is not installed. It is required to load NIfTI files.")
        print("  Install with:  pip install nibabel")
        print()
        resp = input("Install nibabel now? [y/N] ").strip().lower()
        if resp == "y":
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "nibabel"])
            print("[OK] nibabel installed.")
        else:
            print("[WARNING] Continuing without nibabel — NIfTI loading will fail.")
    else:
        import nibabel
        print(f"[OK] nibabel {nibabel.__version__} found.")

    # IXI T2w
    print()
    print("[IXI] Processing IXI T2w dataset (healthy brain, dev tier)...")
    if args.skip_ixi:
        print("  Skipping IXI download (--skip-ixi).")
        ixi_ok = False
    else:
        ixi_ok = download_ixi_t2(args.ixi_dir)

    # BraTS instructions
    print_brats_instructions(args.brats_dir)

    # Summary
    print()
    print("=" * 70)
    print("Next Steps")
    print("=" * 70)
    if ixi_ok:
        print(f"  export IXI_T2_ROOT={args.ixi_dir}")
    else:
        print(f"  export IXI_T2_ROOT=/path/to/ixi_t2   # set after downloading IXI")
    print(f"  export BRATS_ROOT=/path/to/brats2024   # set after downloading BraTS")
    print()
    print("  Then build the benchmark (from the mri/ directory):")
    print("    python build_dataset.py")
    print()
    print("  Without real data, the benchmark builds fine using procedural")
    print("  brain phantoms as fallback.")
    print("=" * 70)


if __name__ == "__main__":
    main()
