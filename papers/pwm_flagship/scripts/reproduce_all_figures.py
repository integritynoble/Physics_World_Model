#!/usr/bin/env python3
"""
Reproduce all main-text figures for the PWM flagship paper.

Usage:
    python reproduce_all_figures.py [--figure N] [--output-dir DIR]

Each figure maps to one or more data-generation scripts and a plotting step.
Run without --figure to reproduce all figures sequentially.

Requirements:
    - Python >= 3.9, PyTorch >= 1.12, CUDA >= 11.3
    - Install: pip install -e packages/pwm_core
    - Datasets: see papers/pwm_flagship/methods.tex, "Datasets" paragraph

Figure-to-Script Map:
    Fig 1 (Overview)          : Schematic; generated in Illustrator/Inkscape
    Fig 2 (OperatorGraph)     : Schematic; generated in Illustrator/Inkscape
    Fig 3 (Triad structure)   : Schematic + heatmap from 4-scenario results
    Fig 4 (Correction bars)   : run_real_data_4scenario.py → plot_fig4.py
    Fig 5 (Deep dives)        : run_real_data_4scenario.py → plot_fig5.py
    Fig 6 (Hardware)          : run_real_data_4scenario.py (real-data mode)
    Fig 7 (Basis growth)      : Static data from solver_registry.yaml
    Tab 1 (Necessity ablation): Static data from FPT paper Proposition 1
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "papers" / "pwm_flagship" / "figures" / "reproduced"


def run_script(script_name, args=None, description=""):
    """Run a script and check for errors."""
    script_path = SCRIPT_DIR / script_name
    if not script_path.exists():
        print(f"  [SKIP] {script_name} not found — see original script for data generation")
        return False
    cmd = [sys.executable, str(script_path)] + (args or [])
    print(f"  [RUN] {description or script_name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [FAIL] {script_name}: {result.stderr[:200]}")
        return False
    print(f"  [OK] {script_name}")
    return True


def reproduce_fig3():
    """Fig 3: Triad structure and gate binding heatmap."""
    print("\n=== Figure 3: Triad Decomposition ===")
    print("  Panel (a): Schematic — manually created")
    print("  Panel (b): Gate binding heatmap — derived from 4-scenario results")
    run_script("run_real_data_4scenario.py",
               description="Generate 4-scenario data for gate binding")


def reproduce_fig4():
    """Fig 4: Cross-modality correction results (centerpiece)."""
    print("\n=== Figure 4: Cross-Modality Correction (Centerpiece) ===")
    print("  Requires: CASSI, CACTI, SPC, Lensless, CT, Ptycho, MRI 4-scenario data")
    run_script("run_real_data_4scenario.py",
               description="CASSI + CACTI + SPC + Lensless 4-scenario")
    run_script("run_ct_4scenario.py",
               description="CT 4-scenario")
    run_script("run_ptycho_4scenario.py",
               description="Ptychography 4-scenario")
    run_script("run_mri_4scenario.py",
               description="MRI 4-scenario")


def reproduce_fig5():
    """Fig 5: Modality deep dives and visual comparison."""
    print("\n=== Figure 5: Deep Dives + Visual Comparison ===")
    run_script("generate_visual_comparison_nature.py",
               description="Generate visual comparison panels")


def reproduce_fig6():
    """Fig 6: Hardware validation on real instruments."""
    print("\n=== Figure 6: Hardware Validation ===")
    run_script("run_real_data_4scenario.py",
               args=["--real-data"],
               description="Real-data residual analysis (CASSI + CACTI)")
    run_script("run_calibration_comparison.py",
               description="Autonomous calibration recovery curves")


def reproduce_fig7():
    """Fig 7: Basis-growth saturation."""
    print("\n=== Figure 7: Basis Growth ===")
    print("  Static plot from solver_registry.yaml modality ordering")
    print("  See generate_all_figures.py for plotting code")
    run_script("generate_all_figures.py",
               description="Generate all static figures including basis growth")


def reproduce_table_necessity():
    """Table 1 (new): Primitive necessity ablation."""
    print("\n=== Table 1: Primitive Necessity Ablation ===")
    print("  Data source: FPT paper Proposition 1 (yang2026fpt)")
    print("  No script needed — values are from the formal proof")
    print("  Witness modalities and epsilon_tier values are in")
    print("  papers/finite_primitive_theorem/main.tex, Proposition 1")


def main():
    parser = argparse.ArgumentParser(description="Reproduce PWM flagship figures")
    parser.add_argument("--figure", type=int, default=None,
                        help="Reproduce only this figure (3-7)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for reproduced figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"PWM Flagship Paper — Figure Reproduction")
    print(f"Output directory: {output_dir}")
    print(f"Repository root:  {REPO_ROOT}")

    figure_map = {
        3: reproduce_fig3,
        4: reproduce_fig4,
        5: reproduce_fig5,
        6: reproduce_fig6,
        7: reproduce_fig7,
    }

    if args.figure:
        if args.figure in figure_map:
            figure_map[args.figure]()
        elif args.figure == 1:
            reproduce_table_necessity()
        else:
            print(f"Figure {args.figure}: Schematic (not scriptable)")
    else:
        reproduce_table_necessity()
        for fig_num in sorted(figure_map):
            figure_map[fig_num]()

    print("\n=== Done ===")
    print("Note: Figures 1-2 are schematics created in Illustrator/Inkscape.")
    print("All data-driven figures use results from the 4-Scenario Protocol.")


if __name__ == "__main__":
    main()
