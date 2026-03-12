#!/usr/bin/env python3
"""Update check.md files for all modalities with CPU test results.

Parses the test results from test_speclab_batch.py output and appends
CPU Algorithm Test Results sections to each modality's check.md.
"""
from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "platform"))

from pwm_platform.services.common_reconstructor import (
    _ensure_challenge_h5,
    _load_sample,
    _dispatch_reconstruction,
    _compute_psnr,
    _compute_ssim,
)
from pwm_platform.services.benchmark_database import get_algorithms, get_variant

CHECK_DIR = Path(__file__).parent.parent / "benchmarks/learn"
TODAY = date.today().isoformat()

_VARIANT_ALIASES = {"cassi": "sd_cassi", "spc": "spc_block"}


def _already_has_cpu_results(check_md: Path) -> bool:
    """Return True if check.md already has a CPU Algorithm Test Results section."""
    try:
        content = check_md.read_text()
        return "## CPU Algorithm Test Results" in content
    except Exception:
        return False


def get_first_cpu_algo(variant_key: str, category: str) -> str | None:
    """Get the first classical/PnP algorithm for a modality."""
    algos = get_algorithms(variant_key, category)
    for a in algos:
        t = a.get("type", "").lower()
        if t in ("classical", "variational", "compressive sensing"):
            return a.get("name", "")
    return algos[0].get("name", "") if algos else None


def run_test(variant_key: str) -> dict | None:
    """Run test on sample_00 only for speed. Returns dict with results or None."""
    import time

    catalog_key = _VARIANT_ALIASES.get(variant_key, variant_key)
    v = get_variant(catalog_key)
    if v is None:
        return None

    category = v.get("category", "compressive")
    algo_name = get_first_cpu_algo(catalog_key, category)
    if not algo_name:
        return None

    try:
        h5_path = _ensure_challenge_h5(variant_key, "public")
        sample = _load_sample(h5_path, sample_idx=0, variant_key=variant_key)
    except Exception as e:
        return {"error": str(e), "algo": algo_name}

    if sample.get("y") is None:
        return {"error": "no measurement data", "algo": algo_name}

    try:
        t0 = time.perf_counter()
        x_recon = _dispatch_reconstruction(sample, variant_key, category, algo_name)
        elapsed = time.perf_counter() - t0
    except Exception as e:
        return {"error": f"reconstruction failed: {e}", "algo": algo_name}

    psnr = None
    ssim = None
    if sample.get("x_true") is not None:
        psnr = _compute_psnr(sample["x_true"], x_recon)
        ssim = _compute_ssim(sample["x_true"], x_recon)

    return {
        "algo": algo_name,
        "psnr": round(psnr, 2) if psnr is not None else None,
        "ssim": round(ssim, 4) if ssim is not None else None,
        "runtime_s": round(elapsed, 2),
    }


def append_cpu_results(check_md: Path, variant_key: str, result: dict) -> None:
    """Append CPU Algorithm Test Results section to check.md."""
    content = check_md.read_text()

    algo = result.get("algo", "Unknown")
    psnr = result.get("psnr")
    ssim = result.get("ssim")
    runtime = result.get("runtime_s", "N/A")
    error = result.get("error")

    if error:
        status = "FAIL"
        metrics_table = f"| Error | {error} |\n"
    else:
        status = "PASS"
        p_str = f"{psnr} dB" if psnr is not None else "N/A"
        s_str = str(ssim) if ssim is not None else "N/A"
        metrics_table = (
            f"| PSNR (sample_00) | {p_str} |\n"
            f"| SSIM (sample_00) | {s_str} |\n"
            f"| Runtime | {runtime} s/sample |\n"
        )

    section = f"""
---

## CPU Algorithm Test Results

**Algorithm:** {algo}
**Type:** Classical CPU
**Test Date:** {TODAY}
**Dataset:** public tier, sample 00
**Status:** {status}

| Metric | Value |
|--------|-------|
{metrics_table}
**Result: {status}**
"""

    with open(check_md, "a") as f:
        f.write(section)


def main():
    modality_dirs = sorted([d for d in CHECK_DIR.iterdir() if d.is_dir()])
    print(f"Updating {len(modality_dirs)} check.md files...")

    skipped = 0
    updated = 0
    failed = 0

    for md in modality_dirs:
        check_file = md / "check.md"
        if not check_file.exists():
            continue
        variant_key = md.name
        if _already_has_cpu_results(check_file):
            skipped += 1
            continue

        print(f"  [{variant_key}] ... ", end="", flush=True)
        result = run_test(variant_key)
        if result is None:
            print("SKIP (no catalog entry)")
            skipped += 1
            continue

        append_cpu_results(check_file, variant_key, result)
        if result.get("error"):
            print(f"FAIL  {result['error']}")
            failed += 1
        else:
            print(f"PASS  PSNR={result.get('psnr')} dB")
            updated += 1

    print(f"\nDone: {updated} updated, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
