# Benchmark QA Check — adaptive_optics

**URL:** https://pwm.platformai.org/benchmark/adaptive_optics
**HTTP Status:** 200
**Check Date:** 2026-03-03 (deep semantic review)

## Summary

| Severity | Count |
|----------|-------|
| HIGH | 3 |
| MEDIUM | 6 |
| LOW | 2 |

---

## HIGH Severity Issues

### H1. Dataset domain mismatch — SEG/EAGE Salt Model attribution
- Citation: "Aminzadeh et al., SEG 1997" — this is a *geophysics* dataset, not AO
- Adaptive optics benchmarks should use astronomical or ophthalmological datasets
- Unclear if this is the correct testbed for AO or a copy-paste error from another modality
**Fix:** Verify dataset is actually AO-relevant; replace citation if incorrect.

### H2. PSNR_norm undefined in scoring formula
- Formula: `0.4 × PSNR_norm + 0.4 × SSIM + 0.2 × consistency`
- Normalization method not specified
- Consistency term uses Ĥ (ideal operator) but mismatch effects should use perturbed model
**Fix:** Define PSNR_norm = (PSNR - baseline) / (max - baseline) with explicit bounds.

### H3. Leaderboard tier aggregation unclear
- Public: Sci-Former 0.803, but Overall: 0.701
- Weighted formula not clearly applied across tiers; cannot verify composite scores
**Fix:** Show composite score calculation example.

---

## MEDIUM Severity Issues

### M1. Forward model DAG missing AO-critical components
DAG shows "M → C → D" but omits: wavefront sensor (WFS), deformable mirror (DM) control loop, temporal servo dynamics. Servo lag is a mismatch parameter but not in DAG.

### M2. AO physics underspecified
- `wfs_centroid_bias` suggests Shack-Hartmann but no subaperture geometry or lenslet spec
- `dm_actuator_gain` lacks actuator count, stroke range, influence functions
- Fried parameter r₀ given as mismatch but is atmospheric turbulence strength — modeling assumptions unclear

### M3. Spec ranges overlap across tiers
`fried_parameter_r0`: Public [0.13, 0.19], Dev [0.126, 0.186], Hidden [0.136, 0.196] — ranges overlap significantly. No clear difficulty progression.

### M4. Missing AO domain references
- No wavefront sensing papers (Shack-Hartmann, pyramid WFS)
- No AO theory (Tyson, Roggemann & Welsh, Hardy)
- No atmospheric turbulence references (Greenwood, Kolmogorov)
- Only 3 method papers cited

### M5. No sample reconstructions or PSF comparisons
PSNR/SSIM require visual validation but no gallery rendered (JavaScript references exist but panels empty).

### M6. HDF5 dataset schema undocumented
Missing: key names, array dimensions, data types, noise model specification.

---

## LOW Severity Issues

### L1. Download button URLs not explicitly shown — unclear if endpoints are active.
### L2. Gallery JavaScript `selectGalleryScene()` references DOM elements not present.

---

*Revised by deep semantic analysis on 2026-03-03.*
