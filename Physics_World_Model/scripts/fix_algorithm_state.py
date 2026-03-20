"""Remove 'done' from algorithm_state.md for solvers with missing modules.

These 33 entries reference modules that don't exist yet (e.g., afm_solvers,
cbct_solvers), so they can't be verified and shouldn't be marked 'done'.
"""

import re
from pathlib import Path

STATE_PATH = Path(r"D:\onedrive\startup\program\physics_world_model\PWM5\Physics_World_Model\datasets\benchmark\algorithm_state.md")

# Algorithms to un-mark (name patterns as they appear in algorithm_state.md)
UNMARK = [
    "AFM-UNet (PWM)",
    "FDK-DL (PWM)",
    "CBCT-UNet (PWM)",
    "CryoCARE (PWM)",
    "DF-UNet (PWM)",
    "DIC-Net (PWM)",
    "DECODE-PAINT (PWM)",
    "EndoMapper-Net (PWM)",
    "AF-SfMLearner (PWM)",
    "EXpansionNet (PWM)",
    "FIB-SEM-Net (PWM)",
    "RETFound (PWM)",
    "DR-Grade-Net (PWM)",
    "ISM-Reassignment-Net (PWM)",
    "LLSM-CARE (PWM)",
    "MFM-UNet (PWM)",
    "MINFLUX-Net (PWM)",
    "NSOM-Net (PWM)",
    "DECODE-SMLM (PWM)",
    "DeepSTORM (PWM)",
    "NeuroLF-PET (PWM)",
    "PET-DL (U-Net) (PWM)",
    "PhaseNet (PWM)",
    "SHG-CARE (PWM)",
    "SPECT-DL (OSEM+) (PWM)",
    "SPECT-UNet (PWM)",
    "SD-CARE (PWM)",
    "STED-Net (CARE) (PWM)",
    "RCAN-STED (PWM)",
    "STM-Net (PWM)",
    "3P-Net (CARE) (PWM)",
    "2P-Net (CARE) (PWM)",
    "2P-DeepInterp (PWM)",
]


def main():
    text = STATE_PATH.read_text(encoding="utf-8")
    lines = text.split("\n")

    count = 0
    for i, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        for algo_name in UNMARK:
            if algo_name in line and "done" in line:
                # Replace the last "done" in the status column with blank
                # The status is in the last data column before the final |
                parts = line.split("|")
                # Find which part has "done"
                for j in range(len(parts) - 1, -1, -1):
                    if "done" in parts[j].strip() and parts[j].strip() == "done":
                        parts[j] = parts[j].replace("done", "")
                        lines[i] = "|".join(parts)
                        count += 1
                        print(f"  Removed 'done' from: {algo_name}")
                        break
                break

    # Update the header count
    # Current: **656/1294 algorithms done (50.7%)**
    new_done = 656 - count
    pct = new_done / 1294 * 100
    old_header = f"**656/1294 algorithms done (50.7%)**"
    new_header = f"**{new_done}/1294 algorithms done ({pct:.1f}%)**"

    for i, line in enumerate(lines):
        if old_header in line:
            lines[i] = line.replace(old_header, new_header)
            print(f"\n  Updated header: {new_header}")
            break

    STATE_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDone: Removed 'done' from {count} unverifiable entries")
    print(f"New count: {new_done}/1294 ({pct:.1f}%)")


if __name__ == "__main__":
    main()
