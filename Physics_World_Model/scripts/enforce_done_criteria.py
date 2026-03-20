#!/usr/bin/env python3
"""Enforce strict done criteria in algorithm_state.md.

Done criteria: PWM_PSNR >= Ref_PSNR - 3.0 dB
- If no Ref_PSNR: not done (can't verify)
- If no PWM_PSNR: not done (no result)
- [proxy] entries: never done
- (test) entries with no ref: not done (no reference to compare)
"""
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
STATE_FILE = ROOT / "datasets" / "benchmark" / "algorithm_state.md"
THRESHOLD = 3.0  # PWM must be within 3 dB of reference

def parse_float(s):
    """Parse a PSNR value, return None if not a number."""
    s = s.strip()
    if not s or s == "—" or s == "–" or s == "-":
        return None
    try:
        return float(s)
    except ValueError:
        return None

def main():
    content = STATE_FILE.read_text(encoding="utf-8")
    lines = content.split("\n")

    kept_done = 0
    removed_done = 0
    removed_entries = []
    kept_entries = []
    current_mod = "unknown"

    for i, line in enumerate(lines):
        # Track current modality
        m = re.match(r"^### \d+\.\s+.*\(`(\w+)`\)", line)
        if m:
            current_mod = m.group(1)

        # Only process table rows with done
        if "| done |" not in line:
            continue

        cols = line.split("|")
        if len(cols) < 10:
            continue
        try:
            int(cols[1].strip())
        except (ValueError, IndexError):
            continue

        algo_name = cols[2].strip()
        ref_psnr_str = cols[5].strip()
        pwm_psnr_str = cols[7].strip()

        ref_psnr = parse_float(ref_psnr_str)
        pwm_psnr = parse_float(pwm_psnr_str)

        keep = False
        reason = ""

        # [proxy] entries: never done
        if "[proxy]" in algo_name:
            keep = False
            reason = "proxy"
        # Both ref and PWM exist: check threshold
        elif ref_psnr is not None and pwm_psnr is not None:
            if pwm_psnr >= ref_psnr - THRESHOLD:
                keep = True
                reason = f"PWM {pwm_psnr:.1f} >= Ref {ref_psnr:.1f} - {THRESHOLD}"
            else:
                keep = False
                reason = f"PWM {pwm_psnr:.1f} < Ref {ref_psnr:.1f} - {THRESHOLD} = {ref_psnr - THRESHOLD:.1f}"
        # No ref PSNR but has PWM PSNR: not done (can't verify against reference)
        elif ref_psnr is None and pwm_psnr is not None:
            keep = False
            reason = f"no ref PSNR (PWM={pwm_psnr:.1f})"
        # No PWM PSNR: not done
        else:
            keep = False
            reason = "no PWM PSNR"

        if keep:
            kept_done += 1
            kept_entries.append(f"  {current_mod}: {algo_name} — {reason}")
        else:
            removed_done += 1
            cols[9] = "  "
            lines[i] = "|".join(cols)
            removed_entries.append(f"  {current_mod}: {algo_name} — {reason}")

    STATE_FILE.write_text("\n".join(lines), encoding="utf-8")

    print(f"algorithm_state.md updated:")
    print(f"  Kept 'done': {kept_done}")
    print(f"  Removed 'done': {removed_done}")
    print()

    if kept_entries:
        print(f"KEPT done ({kept_done} entries):")
        for e in kept_entries[:80]:
            print(e)
        if len(kept_entries) > 80:
            print(f"  ... and {len(kept_entries) - 80} more")

    print()
    if removed_entries:
        print(f"REMOVED done ({removed_done} entries):")
        for e in removed_entries[:100]:
            print(e)
        if len(removed_entries) > 100:
            print(f"  ... and {len(removed_entries) - 100} more")


if __name__ == "__main__":
    main()
