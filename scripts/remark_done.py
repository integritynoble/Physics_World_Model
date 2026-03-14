"""Re-mark 33 algorithms as 'done' now that their solver modules exist."""

import re
from pathlib import Path

STATE_PATH = Path(r"D:\onedrive\startup\program\physics_world_model\PWM5\Physics_World_Model\datasets\benchmark\algorithm_state.md")

REMARK = [
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
        for algo_name in REMARK:
            if algo_name in line and "done" not in line:
                parts = line.split("|")
                # Status is last column before final empty
                # Find the status column (should be mostly empty)
                for j in range(len(parts) - 1, -1, -1):
                    if parts[j].strip() == "" and j > 3:
                        parts[j] = " done "
                        lines[i] = "|".join(parts)
                        count += 1
                        print(f"  Re-marked: {algo_name}")
                        break
                break

    # Update header
    old = "**628/1294 algorithms done (48.5%)**"
    new_done = 628 + count
    pct = new_done / 1294 * 100
    new = f"**{new_done}/1294 algorithms done ({pct:.1f}%)**"

    for i, line in enumerate(lines):
        if old in line:
            lines[i] = line.replace(old, new)
            print(f"\n  Header: {new}")
            break

    STATE_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDone: Re-marked {count} entries, new total: {new_done}")


if __name__ == "__main__":
    main()
