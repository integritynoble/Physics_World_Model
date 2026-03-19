# PWMI-CASSI: Differentiable Calibration for CASSI Forward Model Mismatch

ECCV paper: *Correcting Forward Model Mismatch in Coded Aperture Snapshot Spectral Imaging via Two-Stage Differentiable Calibration*

## Overview

This project demonstrates that operator mismatch (mask shift/rotation) catastrophically degrades deep learning CASSI reconstruction (-16 dB for MST-L), and proposes a two-stage differentiable calibration pipeline that recovers most of this loss.

**Key innovation:** Straight-Through Estimator (STE) for integer dispersion offsets enables gradient-based optimization through the CASSI forward model.

## Quick Start

```bash
# Run complete pipeline (requires GPU, ~4-6 hours)
bash scripts/run_all.sh cuda:0

# Or run individual experiments:
python scripts/validate_cassi_pwmi.py --device cuda:0 --scenes 10
python scripts/run_sensitivity_sweep.py --device cuda:0
python scripts/run_ablation.py --device cuda:0
python scripts/generate_figures.py
python scripts/generate_tables.py
```

## Four-Scenario Protocol

| Scenario | Measurement | Mask | Description |
|----------|------------|------|-------------|
| I (Ideal) | Clean | Ideal | Upper bound |
| II (Assumed) | Corrupted | Ideal | Baseline degradation |
| III (Corrected) | Corrupted | Calibrated (Alg2) | Our method |
| IV (Oracle) | Corrupted | Truth | Oracle recovery |

## File Inventory

```
papers/pwmi_cassi/
├── README.md                          # This file
├── pwmi_cassi_paper.tex               # LaTeX manuscript (ECCV format)
├── llncs.cls                          # LNCS document class
├── splncs04.bst                       # Bibliography style
├── references.bib                     # Bibliography
├── scripts/
│   ├── validate_cassi_pwmi.py         # Main 4-scenario validation (~600 lines)
│   ├── run_sensitivity_sweep.py       # Mismatch magnitude sweep (~300 lines)
│   ├── run_ablation.py                # Alg1-only vs Alg1+Alg2 (~250 lines)
│   ├── generate_figures.py            # Publication figures (~500 lines)
│   ├── generate_tables.py             # LaTeX tables (~200 lines)
│   └── run_all.sh                     # Reproducibility pipeline
├── results/                           # JSON results (populated by scripts)
├── figures/                           # Generated figures (PDF + PNG)
└── tables/                            # Generated LaTeX tables
```

## Experimental Configuration

- **Dataset:** 10 KAIST scenes (256x256x28)
- **Mask:** TSA real mask
- **Mismatch:** dx=1.5 px, dy=1.0 px, theta=0.3 deg
- **Noise:** Poisson (alpha=100000) + Gaussian (sigma=0.01)
- **Methods:** GAP-TV, MST-S, MST-L, HDNet, PnP-HSICNN
- **Dispersion:** s_nom = np.arange(28) * 2 = [0, 2, 4, ..., 54]

## Critical Bug Fix

The previous validation used `s_nom = np.array([2.0] * 28)` (constant array) which collapses all bands to offset 0 after the `dx_f - dx_f.min()` shift inside `DifferentiableCassiForwardSTE`. The correct value is `s_nom = np.arange(28) * 2` (cumulative stride-2 offsets), matching `packages/pwm_core/benchmarks/_cassi_upwmi.py:860`.

## Dependencies

- Python 3.8+
- PyTorch 1.12+ (CUDA recommended)
- NumPy, SciPy, Matplotlib
- pwm_core (from this repository)

## Compile Paper

```bash
pdflatex pwmi_cassi_paper
bibtex pwmi_cassi_paper
pdflatex pwmi_cassi_paper
pdflatex pwmi_cassi_paper
```
