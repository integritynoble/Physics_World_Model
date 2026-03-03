# Benchmark QA Check — afm

**URL:** https://pwm.platformai.org/benchmark/afm
**HTTP Status:** 200
**Check Date:** 2026-03-03 (deep semantic review)

## Summary

| Severity | Count |
|----------|-------|
| HIGH | 4 |
| MEDIUM | 6 |
| LOW | 2 |

---

## HIGH Severity Issues

### H1. Forward model treats AFM as generic imaging — missing cantilever physics
DAG shows only "S → D" (Sampling → Detector). AFM requires:
- Cantilever resonance dynamics (resonance frequency, Q-factor, amplitude setpoint)
- Tip-sample force interaction (van der Waals, adhesion, electrostatic)
- Z-feedback control loop (closed-loop dynamics)
- Force curve approach/retract hysteresis
- Only "tip_shape_convolution" modeled as PSF — loses AFM-specific physics entirely
**Fix:** Add cantilever dynamics, force interaction model, and feedback loop to DAG.

### H2. Gallery section empty — no reconstruction examples visible
JavaScript `selectGalleryScene()` and panel DOM elements exist but no actual images render. Cannot validate dataset realism or reconstruction quality.
**Fix:** Populate gallery with sample AFM reconstructions or add static fallback images.

### H3. Noise model missing cantilever thermal noise
Only generic detector noise listed. AFM-specific noise sources missing:
- Cantilever thermal fluctuations (Brownian motion)
- Electronics noise (detector, amplifier)
- Laser shot noise (for optical lever detection)
**Fix:** Specify noise model with AFM-relevant sources.

### H4. PSNR_norm undefined in scoring formula
Same system-wide issue. No normalization bounds, no worked example.
**Fix:** Define PSNR normalization and show calculation for one method.

---

## MEDIUM Severity Issues

### M1. Spec ranges shift without physical justification
- `piezo_nonlinearity`: Public [-1.0, 2.0] → Hidden [-0.7, 2.3] — negative nonlinearity is unphysical
- `thermal_drift`: asymmetric expansion [-0.14, 0.46] nm/s with no hardware justification
- No reference to typical AFM hardware specs

### M2. Only 3 scenes per tier — insufficient statistical power
No confidence intervals on scores. 3 samples cannot support meaningful PSNR/SSIM statistics.

### M3. Spec primitives are generic imaging (P, M, Π, F, C, Σ, D, S, W, R, Λ)
AFM-specific primitives absent: cantilever model, tip geometry model, force model, feedback loop.

### M4. References incomplete
- "Villarrubia, JRNIST 1997": no exact title/volume/DOI (27 years old)
- "Probe Transformer, 2024": no authors, venue, or DOI
- Alldritt et al. (2020), Zhang et al. (2017): no DOI/URL

### M5. Dataset documentation missing
No disclosure of: sample materials, scan parameters (speed, resolution, setpoint), true mismatch values, measurement noise characteristics.

### M6. HDF5 schema undocumented
Missing: key names, array dimensions, data types.

---

## LOW Severity Issues

### L1. Spec range tables lack units for unitless parameters (piezo_nonlinearity, scanner_hysteresis).
### L2. Spec primitives reference is generic; should include AFM-specific extensions.

---

*Revised by deep semantic analysis on 2026-03-03.*
