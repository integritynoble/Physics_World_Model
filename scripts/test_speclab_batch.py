#!/usr/bin/env python3
"""Batch test SpecLab reconstruction for all modalities.

For each modality:
1. Downloads public HDF5 from GCS (cached)
2. Runs best available CPU algorithm
3. Reports PSNR/SSIM
4. Updates state.md Stage 4 column on success

Usage:
    python3 scripts/test_speclab_batch.py [--start N] [--limit K] [--modality NAME]
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

# Add platform to path
sys.path.insert(0, str(Path(__file__).parent.parent / "platform"))

from pwm_platform.services.common_reconstructor import (
    _dispatch_reconstruction,
    _ensure_challenge_h5,
    _load_sample,
    _compute_psnr,
    _compute_ssim,
    _to_2d_display,
)
from pwm_platform.services.benchmark_database import get_algorithms, get_variant

STATE_MD = Path(__file__).parent.parent / "datasets/benchmark/state.md"

# Best CPU algorithm per modality (override catalog ordering)
_CPU_ALGO_OVERRIDE: dict[str, str] = {
    "ct": "TV-ADMM",
    "cbct": "TV-ADMM",
    "mri": "SENSE",
    "fmri": "SENSE",
    "diffusion_mri": "SENSE",
    "acoustic_emission": "TV-Denoising (Chambolle)",
    "photoacoustic": "Universal Back-Proj",
}


def get_first_cpu_algo(variant_key: str, category: str) -> str | None:
    """Get the first classical/CPU algorithm for a modality."""
    if variant_key in _CPU_ALGO_OVERRIDE:
        return _CPU_ALGO_OVERRIDE[variant_key]
    algos = get_algorithms(variant_key, category)
    for a in algos:
        t = a.get("type", "").lower()
        if t in ("classical", "variational", "compressive sensing", "pnp", "low-rank"):
            return a.get("name", "")
    return algos[0].get("name", "") if algos else None


_VARIANT_ALIASES = {"cassi": "sd_cassi", "spc": "spc_block"}


def test_modality(variant_key: str) -> dict:
    """Test one modality. Returns result dict."""
    result = {"variant": variant_key, "status": "fail", "psnr": None, "ssim": None, "algo": None}

    catalog_key = _VARIANT_ALIASES.get(variant_key, variant_key)
    v = get_variant(catalog_key)
    if v is None:
        result["error"] = "unknown variant"
        return result

    category = v.get("category", "compressive")
    algo_name = get_first_cpu_algo(catalog_key, category)
    if not algo_name:
        result["error"] = "no algorithm found"
        return result
    result["algo"] = algo_name

    try:
        h5_path = _ensure_challenge_h5(variant_key, "public")
        sample = _load_sample(h5_path, sample_idx=0, variant_key=variant_key)
    except Exception as e:
        result["error"] = f"GCS download failed: {e}"
        return result

    if sample.get("y") is None:
        result["error"] = "no measurement data"
        return result

    try:
        t0 = time.perf_counter()
        x_recon = _dispatch_reconstruction(sample, variant_key, category, algo_name)
        elapsed = time.perf_counter() - t0
        result["runtime_s"] = round(elapsed, 2)
    except Exception as e:
        result["error"] = f"reconstruction failed: {e}"
        return result

    if sample.get("x_true") is not None:
        psnr = _compute_psnr(sample["x_true"], x_recon)
        ssim = _compute_ssim(sample["x_true"], x_recon)
        result["psnr"] = round(psnr, 2)
        result["ssim"] = round(ssim, 4)

    result["status"] = "pass"
    return result


def update_state_md(results: list[dict]) -> None:
    """Update state.md Stage 4 column for completed modalities."""
    with open(STATE_MD) as f:
        content = f.read()

    lines = content.split("\n")
    pass_set = {r["variant"] for r in results if r["status"] == "pass"}

    new_lines = []
    for line in lines:
        parts = line.split("|")
        if len(parts) >= 7:
            mod = parts[1].strip()
            if mod in pass_set:
                # parts[6] is Stage 4: SpecLab
                if "❌" in parts[6]:
                    parts[6] = " ✅ "
                    line = "|".join(parts)
        new_lines.append(line)

    content = "\n".join(new_lines)

    # Update summary count
    n_pass = len(pass_set)
    content = re.sub(
        r"Stage 4 \(SpecLab\): \d+/168 ✅",
        f"Stage 4 (SpecLab): {n_pass}/168 ✅",
        content,
    )

    with open(STATE_MD, "w") as f:
        f.write(content)
    print(f"Updated state.md: {n_pass} Stage 4 complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=999)
    parser.add_argument("--modality", type=str, default="")
    parser.add_argument("--update-state", action="store_true", default=True)
    args = parser.parse_args()

    # Get all modalities from state.md
    with open(STATE_MD) as f:
        content = f.read()

    modalities = []
    for line in content.split("\n"):
        parts = line.split("|")
        if len(parts) >= 7 and parts[1].strip() and parts[1].strip() not in ("Modality", "---", ""):
            mod = parts[1].strip()
            stage4 = parts[6].strip() if len(parts) > 6 else ""
            if "✅" not in stage4:  # Only process not-yet-done
                modalities.append(mod)

    if args.modality:
        modalities = [m for m in modalities if args.modality in m]

    modalities = modalities[args.start:args.start + args.limit]
    print(f"Testing {len(modalities)} modalities...")

    results = []
    for i, mod in enumerate(modalities):
        print(f"  [{i+1}/{len(modalities)}] {mod} ... ", end="", flush=True)
        r = test_modality(mod)
        results.append(r)
        if r["status"] == "pass":
            print(f"PASS  algo={r['algo']}  PSNR={r.get('psnr')} dB  t={r.get('runtime_s')}s")
        else:
            print(f"FAIL  {r.get('error', 'unknown error')}")

    if args.update_state:
        # Build cumulative pass set including previously-done modalities
        all_results = [r for r in results if r["status"] == "pass"]
        # Also get already-done modalities from state.md
        for line in content.split("\n"):
            parts = line.split("|")
            if len(parts) >= 7 and parts[1].strip() not in ("Modality", "---", ""):
                mod = parts[1].strip()
                stage4 = parts[6].strip() if len(parts) > 6 else ""
                if "✅" in stage4:
                    all_results.append({"variant": mod, "status": "pass"})
        update_state_md(all_results)

    pass_count = sum(1 for r in results if r["status"] == "pass")
    print(f"\nSummary: {pass_count}/{len(results)} passed")


if __name__ == "__main__":
    main()
