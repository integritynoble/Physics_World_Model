#!/usr/bin/env python3
"""Demonstrate cross-modality transfer for 3+ domain pairs.

Satisfies P3 exit criterion:
  "Cross-modality transfer engine produces actionable suggestions
   for at least 3 domain pairs"
"""
import sys

from pwm_core.science.transfer_engine import CrossModalityTransferEngine


def demo_transfer() -> int:
    te = CrossModalityTransferEngine()
    print(f"Transfer engine loaded: {len(te._modality_signatures)} modality signatures\n")

    targets = ["ct", "mri", "ultrasound", "oct", "pet"]
    total_pairs = 0

    for target in targets:
        suggestions = te.suggest_transfers(target, top_k=3, min_similarity=0.2)
        if suggestions:
            print(f"Transfer suggestions for '{target}':")
            for s in suggestions:
                print(
                    f"  {s.source_modality:20s} -> {target:12s}  "
                    f"similarity={s.confidence:.2f}  "
                    f"shared={s.shared_primitives}"
                )
                total_pairs += 1
            print()

    # ------------------------------------------------------------------ #
    # Validate exit criterion
    # ------------------------------------------------------------------ #
    assert total_pairs >= 3, f"Expected >= 3 pairs, got {total_pairs}"
    print(f"Total actionable transfer pairs: {total_pairs}")
    print("P3 exit criterion satisfied: cross-modality transfer for 3+ domain pairs")
    return 0


if __name__ == "__main__":
    sys.exit(demo_transfer())
