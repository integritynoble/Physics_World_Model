# PWM Flagship Paper — Nature Submission

**Title:** Physics World Models for Computational Imaging: A Universal Physics-Information Law for Recoverability, Carrier Noise, and Operator Mismatch

**Target:** Nature

**Author:** Chengshuai Yang, NextGen PlatformAI C Corp

---

## File Structure

```
papers/pwm_flagship/
  main.tex                    # Main Nature manuscript (~4000 words)
  methods.tex                 # Online Methods (unlimited length)
  supplementary.tex           # Supplementary Information (47 pages, 21 notes)
  pwm_flagship.bib            # Bibliography (~50 references)
  preamble.tex                # Packages and commands
  README.md                   # This file
  figures/                    # Main figures (Fig 1-9, PDF+PNG)
  extended_data/              # Extended Data figures (ED1-ED10)
  original/                   # Original drafts (reference only)
  scripts/                    # Experiment & figure generation scripts
    run_*_4scenario.py        # Single-phantom 4-scenario validation
    run_*_multiphantom.py     # Multi-phantom validation (N=4-5, bootstrap CI)
    aggregate_all_results.py  # Combine all modality results into summary JSON
    generate_all_figures.py   # Nature-quality figure generation (300 DPI)
  results/
    real_data_4scenario/      # Per-modality JSON results
    fluorescence_4scenario/   # Fluorescence experiment results
    combined/                 # Aggregated cross-modality summary
```

## Building

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Reproducing Multi-Phantom Experiments

All experiments run on CPU (no GPU required):

```bash
cd /path/to/Physics_World_Model
export PYTHONPATH=packages/pwm_core:$PYTHONPATH

# Run individual modality experiments
python3 papers/pwm_flagship/scripts/run_cryoem_multiphantom.py      # ~40s, N=5
python3 papers/pwm_flagship/scripts/run_fluorescence_multiphantom.py # ~40s, N=5
python3 papers/pwm_flagship/scripts/run_ultrasound_multiphantom.py   # ~100s, N=5
python3 papers/pwm_flagship/scripts/run_compholo_multiphantom.py     # ~120s, N=4
python3 papers/pwm_flagship/scripts/run_cbct_multiphantom.py         # ~15min, N=5

# Aggregate all results
python3 papers/pwm_flagship/scripts/aggregate_all_results.py

# Regenerate figures
python3 papers/pwm_flagship/scripts/generate_all_figures.py
```

## Key Numbers

| Metric | Value |
|--------|-------|
| Modalities compiled | 168 (across 19 categories) |
| Unique DAG patterns | 37 |
| OperatorGraph templates | 168 |
| Physical carriers | 5 families (photons, electrons, spins, acoustic, X-rays) |
| Modalities fully validated (Phase 1) | 9 (single-phantom) |
| Modalities validated (Phase 2) | 5 (multi-phantom, bootstrap CI) |
| Total validated configurations | 14 |
| Correction range | +0.2 to +48.25 dB |
| Gate 3 dominant | 14/14 configurations (100%) |
| 168-modality registry | All covered by 11 primitives |

### Phase 2 Multi-Phantom Results (5 carrier families)

| Modality | Carrier | N | Gate 3 Parameter | Max Delta (dB) | Recovery |
|----------|---------|---|------------------|----------------|----------|
| Fluorescence | Photon | 5 | PSF sigma | +8.35 +/- 3.58 | 0.53 |
| CT (offset) | X-ray | 5 | Detector offset | +6.53 +/- 1.82 | 1.000 |
| Cryo-EM | Electron | 5 | Defocus | +3.30 +/- 1.03 | 1.000 |
| Comp. Holography | Photon | 4 | Prop. distance | +1.04 +/- 0.48 | 1.10 |
| Ultrasound | Acoustic | 5 | Speed of sound | +0.20 +/- 0.10 | --- |

## Main Figures

1. **PWM Overview** — End-to-end pipeline from 168 modalities to 11 primitives
2. **OperatorGraph IR** — DAG examples + Physics Fidelity Ladder (4 tiers)
3. **Triad Law** — Decision tree + gate binding heatmap across carriers
4. **14-Configuration Correction** — Bar chart across 5 carrier families (Phase 1 + Phase 2)
5. **CASSI/CACTI Deep Dive** — 4-scenario comparison with state-of-the-art solvers
6. **Zero-Shot Generalization** — Cross-carrier transfer across 12 modalities
7. **Hardware Validation** — Real CASSI + CACTI instrument results
8. **Visual Comparison** — Reconstruction quality across scenarios
9. **Basis Growth** — Primitive count vs modality coverage
