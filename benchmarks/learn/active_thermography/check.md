# Benchmark QA Check — active_thermography

**URL:** https://pwm.platformai.org/benchmark/active_thermography
**HTTP Status:** 200
**Check Date:** 2026-03-03 (deep semantic review)

## Summary

| Severity | Count |
|----------|-------|
| HIGH | 3 |
| MEDIUM | 7 |
| LOW | 2 |

---

## HIGH Severity Issues

### H1. Forward model DAG missing thermal diffusion physics
DAG shows only "P → D" but active thermography requires:
- Time-domain heat equation (thermal PDE)
- Thermal diffusivity parameters (k, ρ, c)
- Excitation waveform (modulation frequency, pulse shape)
- Thermal boundary conditions
- No temporal integration primitive shown
**Fix:** Expand DAG to include Σ (temporal integration), add thermal PDE, specify if lock-in or transient thermography.

### H2. PSNR_norm undefined and Ĥ vs H notation inconsistent
- Scoring: `0.4 × PSNR_norm + 0.4 × SSIM + 0.2 × (1 − ‖y − Ĥx̂‖/‖y‖)`
- PSNR_norm normalization method/range not specified
- Formula uses Ĥ (estimated operator) but description says "ideal forward operator (H)"
- No weighting justification for 40/40/20 split
**Fix:** Define PSNR_norm bounds, use H consistently, justify weights.

### H3. References incomplete — DAGM 2007 citation has no DOI/URL
- "Wieler & Hahn, DAGM 2007" — no full citation, DOI, or download link
- NDT-Former, DefectNet papers have no arXiv or venue info
**Fix:** Provide DOIs for all citations.

---

## MEDIUM Severity Issues

### M1. Cross-tier ranking reversal unexplained
PnP-DnCNN surpasses DefectNet on Dev despite lower Public score. No discussion of why.

### M2. Dev tier scoring contradicts blind evaluation
"Blind — no ground truth" but PSNR/SSIM require ground truth. Server-side scoring not explicitly stated.

### M3. Spec ranges artificially narrow
- emissivity_error: 0.94–0.97 (±3%) — real variation can be much larger
- background_temperature: 23.8–27.3°C (3.5°C span) — real environments vary more
- No sensitivity analysis or justification against real datasets

### M4. Fresnel/Rayleigh-Sommerfeld kernels listed but irrelevant
These propagation kernels are for wave optics/acoustics, not thermal diffusion. Listed as "kernel" options but wrong physics domain.

### M5. HDF5 dataset schema undocumented
Missing: key names, dimensions, dtypes, noise model. No example loading code.

### M6. Hidden tier Docker submission specs missing
No base image, entrypoint signature, runtime limits, or output format specified.

### M7. Missing defect-level metrics
PSNR/SSIM optimize for pixel intensity but don't measure defect detection accuracy. Should include precision, recall, IoU, or Dice coefficient.

---

## LOW Severity Issues

### L1. Gallery JavaScript references unfulfilled — `selectGalleryScene()` panels don't render.
### L2. Phase information ignored — lock-in thermography uses phase images but only intensity metrics evaluated.

---

*Revised by deep semantic analysis on 2026-03-03.*
