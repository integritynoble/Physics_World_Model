# PWM Flagship Paper — Nature Submission

**Title:** Physics World Models for Computational Imaging: A Universal Physics-Information Law for Recoverability, Carrier Noise, and Operator Mismatch

**Target:** Nature

**Author:** Chengshuai Yang, NextGen PlatformAI C Corp

---

## File Structure

```
papers/pwm_flagship/
  main.tex                  # Main Nature manuscript (~4000 words)
  methods.tex               # Online Methods (unlimited length)
  supplementary.tex         # Supplementary Information
  pwm_flagship.bib          # Bibliography (~50 references)
  preamble.tex              # Packages and commands
  README.md                 # This file
  figures/                  # Main figures (Fig 1-6)
  extended_data/            # Extended Data figures (ED1-ED10)
  original/                 # Original drafts (reference only)
    PWM_Nature_Paper_Draft_v1.pdf
    PWM Nature Manuscript.pdf
    *.docx
```

## Building

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Key Numbers

| Metric | Value |
|--------|-------|
| Modalities compiled | 64 |
| OperatorGraph templates | 89 |
| Physical carriers | 5 (photons, electrons, spins, acoustic, particles) |
| Modalities fully validated | 16 |
| Correction range | +0.8 to +48.25 dB |
| Median correction gain | +14.5 dB |
| Gate 3 dominant | 14/16 modalities (87.5%) |
| Worst mismatch degradation | -16.72 dB (MST-L, CASSI) |
| 26-modality benchmark | All PASS |

## Main Figures

1. **PWM Overview** — End-to-end pipeline
2. **OperatorGraph IR** — DAG examples + Fidelity Ladder
3. **Triad Law** — Decision tree + gate binding heatmap
4. **16-Modality Correction** — Bar chart by carrier family
5. **CASSI/CACTI Deep Dive** — 4-scenario comparison
6. **Zero-Shot Generalization** — Cross-carrier transfer
