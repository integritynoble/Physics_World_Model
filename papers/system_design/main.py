"""
Multi-Agent Imaging System Design Pipeline
==========================================

Usage:
  python main.py --modality ct --period forward \
      --prompt "Design a sparse-view CT system with 60 angles and low dose"

  python main.py --modality mri --period reconstruction \
      --prompt "Reconstruct 4x accelerated Cartesian k-space with motion correction"

  python main.py --modality widefield --period forward \
      --prompt "Design a live-cell widefield system with minimal phototoxicity"

Outputs:
  outputs/{modality}_{period}_v1_iter{n}.md   — plan markdown
  outputs/{modality}_{period}_result.npz      — measurements / reconstruction arrays
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np

# Support both: `python3 main.py` (from within system_design/) and
# `python3 -m system_design.main` (from papers/ parent).
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from system_design.pipeline.orchestrator import Orchestrator
from system_design.agents.performance_agent import ForwardResult, ReconResult
from system_design.config import OUTPUT_DIR, PSNR_TARGET, SSIM_TARGET
from system_design.database import list_modalities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-agent imaging system design pipeline"
    )
    parser.add_argument(
        "--modality", required=True,
        help=f"Imaging modality. Known: {', '.join(list_modalities())}"
    )
    parser.add_argument(
        "--period", required=True, choices=["forward", "reconstruction"],
        help="'forward' = design the imaging system; 'reconstruction' = design the algorithm"
    )
    parser.add_argument(
        "--prompt", required=True,
        help="Natural language description of the design task"
    )
    parser.add_argument(
        "--phantom", default=None,
        help="Path to .npy phantom file (forward period only, optional)"
    )
    parser.add_argument(
        "--measurements", default=None,
        help="Path to .npy measurements file (reconstruction period, required)"
    )
    parser.add_argument(
        "--x_true", default=None,
        help="Path to .npy ground truth file (reconstruction, optional for metrics)"
    )
    parser.add_argument(
        "--no-verbose", action="store_true",
        help="Suppress progress output"
    )
    return parser.parse_args()


def load_npy(path: str | None) -> np.ndarray | None:
    if path is None:
        return None
    arr = np.load(path)
    print(f"  Loaded {path}: shape={arr.shape} dtype={arr.dtype}")
    return arr


def save_results(result: ForwardResult | ReconResult, modality: str, period: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fname = os.path.join(OUTPUT_DIR, f"{modality}_{period}_result.npz")

    if isinstance(result, ForwardResult):
        np.savez(
            fname,
            measurements=result.measurements,
            ground_truth=result.ground_truth,
        )
        print(f"\nSaved forward result → {fname}")
        print(f"  measurements shape: {result.measurements.shape}")
        print(f"  ground_truth shape: {result.ground_truth.shape}")
    else:
        save_dict = {"x_hat": result.x_hat}
        if result.x_true is not None:
            save_dict["x_true"] = result.x_true
        np.savez(fname, **save_dict)
        print(f"\nSaved reconstruction result → {fname}")
        if result.psnr_db is not None:
            status = "PASS" if (result.psnr_db >= PSNR_TARGET and
                                result.ssim_val >= SSIM_TARGET) else "below target"
            print(f"  PSNR={result.psnr_db:.2f} dB  SSIM={result.ssim_val:.4f}  [{status}]")
            print(f"  Targets: PSNR≥{PSNR_TARGET} dB, SSIM≥{SSIM_TARGET}")


def main() -> int:
    args = parse_args()
    verbose = not args.no_verbose

    print(f"\nMulti-Agent System Design Pipeline")
    print(f"  Modality : {args.modality}")
    print(f"  Period   : {args.period}")
    print(f"  Prompt   : {args.prompt[:80]}...")

    phantom      = load_npy(args.phantom)
    measurements = load_npy(args.measurements)
    x_true       = load_npy(args.x_true)

    orchestrator = Orchestrator()
    result = orchestrator.run(
        modality=args.modality,
        period=args.period,
        prompt=args.prompt,
        phantom=phantom,
        measurements=measurements,
        x_true=x_true,
        verbose=verbose,
    )

    print(f"\n{'='*60}")
    print(f"Pipeline finished: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Plan: {result.plan.markdown_path}")

    if not result.success:
        print(f"  Failure: {result.failure_reason}")
        return 1

    if result.performance is not None:
        save_results(result.performance, args.modality, args.period)

    return 0


if __name__ == "__main__":
    sys.exit(main())
