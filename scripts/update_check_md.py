#!/usr/bin/env python3
"""Update check.md files for all modalities with CPU test results.

Tests ALL CPU-type algorithms per modality (not just the first one), so that
gen_speclab_state.py can accurately verify each algorithm individually.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
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

_CPU_TYPES = {
    "classical", "variational", "compressive sensing", "pnp", "low-rank",
    "plug-and-play", "dictionary learning", "compressive sensing",
}


def is_cpu_type(algo_type: str) -> bool:
    t = algo_type.lower().strip()
    return any(c in t for c in _CPU_TYPES)


def _algo_name_matches(a: str, b: str) -> bool:
    """Fuzzy-match: normalize and compare."""
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())
    return norm(a) == norm(b)


def _has_result_for_algo(check_md: Path, algo_name: str) -> bool:
    """Return True if check.md already has a CPU result for this specific algorithm."""
    try:
        content = check_md.read_text()
    except Exception:
        return False
    for m in re.finditer(r"\*\*Algorithm:\*\*\s*(.+)", content):
        if _algo_name_matches(m.group(1).strip(), algo_name):
            return True
    return False


def get_cpu_algos(variant_key: str, category: str) -> list[str]:
    """Return all CPU-type algorithm names for a modality."""
    algos = get_algorithms(variant_key, category)
    return [a["name"] for a in algos if is_cpu_type(a.get("type", ""))]


def run_test(variant_key: str, algo_name: str, sample: dict, category: str) -> dict:
    """Run one algorithm on the pre-loaded sample. Returns result dict."""
    try:
        t0 = time.perf_counter()
        x_recon = _dispatch_reconstruction(sample, variant_key, category, algo_name)
        elapsed = time.perf_counter() - t0
    except Exception as e:
        return {"algo": algo_name, "error": f"reconstruction failed: {e}"}

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


def append_cpu_result(check_md: Path, result: dict) -> None:
    """Append a CPU Algorithm Test Results section to check.md."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", type=str, default="", help="Filter to specific modality")
    parser.add_argument("--force", action="store_true", help="Re-test even if already in check.md")
    parser.add_argument("--limit", type=int, default=999, help="Max modalities to process")
    args = parser.parse_args()

    modality_dirs = sorted([d for d in CHECK_DIR.iterdir() if d.is_dir()])
    if args.modality:
        modality_dirs = [d for d in modality_dirs if args.modality in d.name]
    modality_dirs = modality_dirs[:args.limit]
    print(f"Processing {len(modality_dirs)} check.md files (all CPU algorithms)...")

    total_tested = 0
    total_skipped = 0
    total_failed = 0

    for md in modality_dirs:
        check_file = md / "check.md"
        if not check_file.exists():
            continue

        variant_key = md.name
        catalog_key = _VARIANT_ALIASES.get(variant_key, variant_key)
        v = get_variant(catalog_key)
        if v is None:
            continue

        category = v.get("category", "compressive")
        cpu_algos = get_cpu_algos(catalog_key, category)
        if not cpu_algos:
            continue

        # Filter to untested algorithms (or all if --force)
        to_test = []
        for algo in cpu_algos:
            if args.force or not _has_result_for_algo(check_file, algo):
                to_test.append(algo)
            else:
                total_skipped += 1

        if not to_test:
            continue

        # Load sample once for all algorithms in this modality
        try:
            h5_path = _ensure_challenge_h5(variant_key, "public")
            sample = _load_sample(h5_path, sample_idx=0, variant_key=variant_key)
        except Exception as e:
            print(f"  [{variant_key}] SKIP — load failed: {e}")
            total_skipped += len(to_test)
            continue

        if sample.get("y") is None:
            print(f"  [{variant_key}] SKIP — no measurement data")
            total_skipped += len(to_test)
            continue

        for algo in to_test:
            print(f"  [{variant_key}] {algo} ... ", end="", flush=True)
            result = run_test(variant_key, algo, sample, category)
            append_cpu_result(check_file, result)
            if result.get("error"):
                print(f"FAIL  {result['error']}")
                total_failed += 1
            else:
                print(f"PASS  PSNR={result.get('psnr')} dB  SSIM={result.get('ssim')}  t={result.get('runtime_s')}s")
                total_tested += 1

    print(f"\nDone: {total_tested} passed, {total_skipped} skipped, {total_failed} failed")


if __name__ == "__main__":
    main()
