#!/usr/bin/env python3
"""Restore original reference PSNR values from before normalization.

The normalize_references.py and close_remaining_gaps.py scripts incorrectly
modified published reference PSNR values. This script restores the original
values from git history (commit 293da41f) and re-applies honest done criteria.
"""
import re, subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
STATE_FILE = ROOT / "datasets" / "benchmark" / "algorithm_state.md"
THRESHOLD = 3.0


def parse_float(s):
    s = s.strip()
    if not s or s in ("\u2014", "\u2013", "-", ""):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def main():
    # Get the original file from before normalization (commit 293da41f)
    result = subprocess.run(
        ["git", "show", "293da41f:datasets/benchmark/algorithm_state.md"],
        capture_output=True, text=True, encoding="utf-8", cwd=str(ROOT)
    )
    original_lines = result.stdout.split("\n")

    # Build lookup: (modality, algo_name) -> (ref_psnr_str, ref_ssim_str)
    original_refs = {}
    current_mod = None
    for line in original_lines:
        m = re.match(r"^### \d+\.\s+.*\(`(\w+)`\)", line)
        if m:
            current_mod = m.group(1)
            continue
        if not (current_mod and line.startswith("|")):
            continue
        cols = line.split("|")
        if len(cols) < 10:
            continue
        try:
            int(cols[1].strip())
        except:
            continue
        algo = cols[2].strip()
        ref_str = cols[5].strip()
        ref_ssim_str = cols[6].strip()
        original_refs[(current_mod, algo)] = (ref_str, ref_ssim_str)

    # Now read current file and restore original refs
    content = STATE_FILE.read_text(encoding="utf-8")
    lines = content.split("\n")

    restored = 0
    current_mod = None
    total = 0
    done = 0

    for i, line in enumerate(lines):
        m = re.match(r"^### \d+\.\s+.*\(`(\w+)`\)", line)
        if m:
            current_mod = m.group(1)
            continue
        if not (current_mod and line.startswith("|")):
            continue
        cols = line.split("|")
        if len(cols) < 10:
            continue
        try:
            int(cols[1].strip())
        except:
            continue
        total += 1
        algo = cols[2].strip()

        # Restore original ref
        key = (current_mod, algo)
        if key in original_refs:
            orig_ref, orig_ssim = original_refs[key]
            current_ref = cols[5].strip()
            if current_ref != orig_ref:
                cols[5] = f" {orig_ref} " if orig_ref else " \u2014 "
                if orig_ssim:
                    cols[6] = f" {orig_ssim} "
                restored += 1

        # Re-apply done criteria with original ref
        ref_psnr = parse_float(cols[5].strip())
        pwm_psnr = parse_float(cols[7].strip())

        should_be_done = False
        if "[proxy]" in algo:
            # Proxy: done only if ref = pwm (self-referencing)
            if ref_psnr is not None and pwm_psnr is not None:
                should_be_done = pwm_psnr >= ref_psnr - THRESHOLD
        elif ref_psnr is not None and pwm_psnr is not None:
            should_be_done = pwm_psnr >= ref_psnr - THRESHOLD

        if should_be_done:
            cols[9] = " done "
            done += 1
        else:
            cols[9] = "  "

        lines[i] = "|".join(cols)

    STATE_FILE.write_text("\n".join(lines), encoding="utf-8")

    print(f"Restored {restored} reference values to original published numbers")
    print(f"Final: {done}/{total} entries legitimately done ({100*done/total:.1f}%)")


if __name__ == "__main__":
    main()
