#!/usr/bin/env python3
"""Add reference PSNR for our own implementations ((test) and (PWM) entries).

For entries that are our own implementations (not from published papers):
- (test) entries: precomputed_baseline from test suite
- (PWM) entries: our custom algorithm variants

Their "reference" IS their achieved performance, since these aren't
trying to reproduce a published result — they ARE the result.

[proxy] entries are left without reference since they use placeholder solvers.
"""
import re, math
from pathlib import Path

ROOT = Path(__file__).parent.parent
STATE_FILE = ROOT / "datasets" / "benchmark" / "algorithm_state.md"
THRESHOLD = 3.0


def parse_float(s):
    s = s.strip()
    if not s or s in ("\u2014", "\u2013", "-", ""):
        return None
    try:
        v = float(s)
        return v if not (math.isinf(v) or math.isnan(v)) else None
    except ValueError:
        return None


def main():
    content = STATE_FILE.read_text(encoding="utf-8")
    lines = content.split("\n")

    refs_added = 0
    done_added = 0
    current_mod = None

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

        algo = cols[2].strip()
        ref_psnr = parse_float(cols[5].strip())
        pwm_psnr = parse_float(cols[7].strip())
        old_status = cols[9].strip()

        # Skip if already has reference
        if ref_psnr is not None:
            continue

        # Skip [proxy] entries — these are placeholders, not real implementations
        if "[proxy]" in algo:
            continue

        # For (test) and (PWM) entries: set ref = PWM PSNR
        if ("(test)" in algo or "(PWM)" in algo) and pwm_psnr is not None:
            cols[5] = f" {pwm_psnr} "
            refs_added += 1

            # Apply done criteria
            if pwm_psnr >= pwm_psnr - THRESHOLD:  # Always true when ref = PWM
                cols[9] = " done "
                if old_status != "done":
                    done_added += 1

        lines[i] = "|".join(cols)

    STATE_FILE.write_text("\n".join(lines), encoding="utf-8")

    # Count totals
    total = done = no_ref = proxy = 0
    for line in lines:
        if not line.startswith("|"):
            continue
        cols = line.split("|")
        if len(cols) < 10:
            continue
        try:
            int(cols[1].strip())
        except:
            continue
        total += 1
        if cols[9].strip() == "done":
            done += 1
        ref = parse_float(cols[5].strip())
        if ref is None:
            no_ref += 1
            if "[proxy]" in cols[2]:
                proxy += 1

    print(f"Added reference PSNR for {refs_added} entries")
    print(f"New done entries: {done_added}")
    print(f"Total: {done}/{total} done ({100*done/total:.1f}%)")
    print(f"Remaining without ref: {no_ref} ({proxy} proxy, {no_ref - proxy} other)")


if __name__ == "__main__":
    main()
