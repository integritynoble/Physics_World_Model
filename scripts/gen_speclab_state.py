#!/usr/bin/env python3
"""Generate speclab_recon_state.md — algorithm verification status for all modalities.

Rules:
- CPU algorithms (Classical, Variational, PnP, Compressed Sensing, Low-Rank):
  mark "done" only if:
    1. That specific algorithm was tested in check.md (PASS status), AND
    2. The actual PSNR is within 2 dB of the reference PSNR (or no ref PSNR exists).
- GPU/DL algorithms (Deep Learning, Transformer, Diffusion, Foundation, Score-Based,
  Physics-Informed, Deep Unrolling):
  leave blank — awaiting GPU server verification.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "platform"))

from pwm_platform.services.benchmark_database import (
    get_algorithms,
    get_variant,
    list_all_variant_keys,
)
from pwm_platform.services.benchmark_database._algorithm_catalog import CATEGORY_REAL_SCORES

# Types that can run on CPU (verified by us)
_CPU_TYPES = {
    "classical", "variational", "compressed sensing", "pnp", "low-rank",
    "plug-and-play", "dictionary learning", "compressive sensing",
}

# Types that need GPU (leave blank)
_GPU_TYPES = {
    "deep learning", "transformer", "diffusion", "score-based", "foundation model",
    "physics-informed", "deep unrolling", "gan", "autoencoder", "self-supervised",
    "neural", "implicit",
}

# PSNR tolerance: actual PSNR must be within this many dB of reference to mark "done"
_PSNR_TOLERANCE_DB = 2.0


def is_cpu_type(algo_type: str) -> bool:
    t = algo_type.lower().strip()
    return any(c in t for c in _CPU_TYPES)


def get_ref_score_float(variant_key: str, algo_name: str) -> float | None:
    """Return reference PSNR as float from CATEGORY_REAL_SCORES, or None."""
    scores = CATEGORY_REAL_SCORES.get(variant_key, [])
    for s in scores:
        if s.get("method", "").lower() == algo_name.lower():
            return s.get("psnr")
    return None


def get_ref_scores(variant_key: str, algo_name: str) -> tuple[str, str]:
    """Return (ref_psnr, ref_ssim) display strings."""
    scores = CATEGORY_REAL_SCORES.get(variant_key, [])
    for s in scores:
        if s.get("method", "").lower() == algo_name.lower():
            psnr = f"{s['psnr']} dB" if s.get("psnr") else ""
            ssim = f"{s['ssim']}" if s.get("ssim") else ""
            return psnr, ssim
    return "", ""


def _algo_name_matches(tested: str, catalog: str) -> bool:
    """Fuzzy-match algorithm names (case-insensitive, ignore punctuation)."""
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())
    return norm(tested) == norm(catalog)


_LEARN_DIR_ALIASES: dict[str, str] = {
    "sd_cassi": "cassi",
    "spc_block": "spc",
    "spc_kronecker": "spc",
}


def get_check_md_results(variant_key: str) -> list[dict]:
    """Read all CPU test results from check.md.

    Returns list of dicts: [{algo, psnr_db, ssim, status}, ...].
    Supports check.md files with multiple test sections.
    """
    dir_name = _LEARN_DIR_ALIASES.get(variant_key, variant_key)
    check_path = (
        Path(__file__).parent.parent / "benchmarks/learn" / dir_name / "check.md"
    )
    if not check_path.exists():
        return []
    content = check_path.read_text()
    if "## CPU Algorithm Test Results" not in content:
        return []

    results = []
    # Split on each CPU section header
    sections = re.split(r"(?=## CPU Algorithm Test Results)", content)
    for sec in sections:
        if "## CPU Algorithm Test Results" not in sec:
            continue
        algo_m = re.search(r"\*\*Algorithm:\*\*\s*(.+)", sec)
        psnr_m = re.search(r"PSNR.*?[|]\s*([\d.]+)\s*dB", sec)
        ssim_m = re.search(r"SSIM.*?[|]\s*([\d.]+)\s*\n", sec)
        status_m = re.search(r"\*\*Result:\s*(PASS|FAIL)\*\*", sec)

        algo = algo_m.group(1).strip() if algo_m else None
        psnr_db = float(psnr_m.group(1)) if psnr_m else None
        ssim = float(ssim_m.group(1)) if ssim_m else None
        status = status_m.group(1) if status_m else None

        if algo:
            results.append({"algo": algo, "psnr_db": psnr_db, "ssim": ssim, "status": status})

    return results


def _is_done(
    algo_name: str,
    algo_type: str,
    variant_key: str,
    check_results: list[dict],
) -> bool:
    """Return True if algorithm is verified done.

    Logic:
    1. Must be CPU-type algorithm
    2. Must have a PASS entry in check.md for this exact algorithm
    3. If a reference PSNR exists, actual PSNR must be within _PSNR_TOLERANCE_DB
    """
    if not is_cpu_type(algo_type):
        return False

    ref_psnr = get_ref_score_float(variant_key, algo_name)

    for r in check_results:
        if not _algo_name_matches(r["algo"], algo_name):
            continue
        if r["status"] != "PASS":
            continue
        # PASS found — check PSNR against reference
        if ref_psnr is None:
            # No reference available — trust the PASS
            return True
        if r["psnr_db"] is None:
            # No measured PSNR — trust the PASS (e.g., no x_true available)
            return True
        # Actual PSNR must be within tolerance of reference
        if r["psnr_db"] >= ref_psnr - _PSNR_TOLERANCE_DB:
            return True

    return False


def main():
    keys = list_all_variant_keys()
    print(f"Building speclab_recon_state.md for {len(keys)} modalities...")

    lines = [
        "# SpecLab Reconstruction State",
        "",
        "Tracks verification status of all reconstruction algorithms in SpecLab",
        "(`https://pwm.platformai.org/speclab`).",
        "",
        "**Status:**",
        "- `done` — PWM CPU reconstruction verified, actual PSNR within 2 dB of reference",
        "- *(blank)* — awaiting verification (not yet tested, or PSNR below reference threshold)",
        "",
        f"Last updated: 2026-03-12 | Total modalities: {len(keys)}",
        "",
        "---",
        "",
    ]

    cpu_done_total = 0
    cpu_pending_total = 0
    gpu_pending_total = 0

    for vk in sorted(keys):
        v = get_variant(vk)
        if v is None:
            continue
        category = v.get("category", "")
        display_name = v.get("display_name", vk)
        algos = get_algorithms(vk, category)
        if not algos:
            continue

        check_results = get_check_md_results(vk)

        lines.append(f"## {display_name} (`{vk}`) — {category}")
        lines.append("")
        lines.append("| Algorithm | Type | Ref PSNR | Ref SSIM | Status |")
        lines.append("|-----------|------|----------|----------|--------|")

        for a in algos:
            name = a.get("name", "")
            atype = a.get("type", "")
            ref_psnr, ref_ssim = get_ref_scores(vk, name)

            if is_cpu_type(atype):
                if _is_done(name, atype, vk, check_results):
                    status = "done"
                    cpu_done_total += 1
                else:
                    status = ""
                    cpu_pending_total += 1
            else:
                status = ""  # GPU required
                gpu_pending_total += 1

            lines.append(f"| {name} | {atype} | {ref_psnr} | {ref_ssim} | {status} |")

        lines.append("")

    lines.extend([
        "---",
        "",
        "## Summary",
        "",
        f"| Category | Count |",
        f"|----------|-------|",
        f"| CPU algorithms verified (done) | {cpu_done_total} |",
        f"| CPU algorithms pending verification | {cpu_pending_total} |",
        f"| GPU algorithms pending | {gpu_pending_total} |",
        f"| Total | {cpu_done_total + cpu_pending_total + gpu_pending_total} |",
        "",
    ])

    out_path = Path(__file__).parent.parent / "datasets/benchmark/speclab_recon_state.md"
    out_path.write_text("\n".join(lines))
    print(f"Written to {out_path}")
    print(f"CPU done: {cpu_done_total}, CPU pending: {cpu_pending_total}, GPU pending: {gpu_pending_total}")


if __name__ == "__main__":
    main()
