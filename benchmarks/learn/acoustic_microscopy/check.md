# Benchmark QA Check — acoustic_microscopy

**URL:** https://pwm.platformai.org/benchmark/acoustic_microscopy
**HTTP Status:** 200
**Check Date:** 2026-03-03 (deep semantic review)

## Summary

| Severity | Count |
|----------|-------|
| HIGH | 2 |
| MEDIUM | 8 |
| LOW | 3 |

---

## HIGH Severity Issues

### H1. Leaderboard ranking inconsistency
- NDT-Former ranks #1 overall (0.736) but DefectNet ranks #2 on Public (0.764) while ranking #3 overall (0.626)
- Check: `0.4(0.797) + 0.4(0.748) + 0.2(0.664) = 0.7434 ≈ 0.736` — tier weighting partially reconciles but method isn't transparent
- DefectNet's 0.626 overall from 0.764/0.597/0.517 tier scores needs verification
**Fix:** Show composite score formula worked example for every method; verify arithmetic.

### H2. Spec range variance methodology contradictory
- Dev/Hidden are "blind evaluation" but exact parameter ranges are published on the page
- If blind, ranges should be withheld; if known, why vary them?
- Focus_depth_error: Public [-4, 8], Dev [-4.8, 7.2], Hidden [-2.8, 9.2] — no nesting pattern
**Fix:** Clarify if ranges are revealed post-submission or withheld; ensure proper difficulty progression.

---

## MEDIUM Severity Issues

### M1. Forward model DAG incomplete
DAG shows only `P → D` (Propagation → Detector). Missing: transducer excitation/bandwidth model, beam forming, frequency-dependent attenuation, time-gating. Should be `[S] → P → [C] → D`.

### M2. PSNR_norm undefined in scoring formula
Same system-wide issue — normalization method, range, bounds not specified.

### M3. Consistency term lacks noise-floor awareness
`(1 − ‖y − Ĥx̂‖/‖y‖)` penalizes residual but noise η makes this non-zero even for perfect reconstruction.

### M4. Gate position error units missing
Listed as "−" (dash). Should specify "ns" or "sample indices" with physical time conversion.

### M5. HDF5 dataset format unspecified
Missing: key names, signal dimensions (1D vs 2D), data types (float32?), measurement noise model.

### M6. Dev tier scoring method unclear
Dev has no visible ground truth — how are PSNR/SSIM computed? Server-side ground truth not explicitly stated.

### M7. "Nominal" vs "Perturbed" column ambiguity
Single perturbed values shown (e.g., coupling_medium_speed: 1494.0) instead of range — unclear if representative or fixed.

### M8. Incomplete references
- "Wieler & Hahn, DAGM 2007": no full citation, no DOI
- "NDT Transformer, 2024": no author, no arXiv link
- "U-Net for NDT, 2021": no venue
- TSR (Shepard et al. 2003) unlabeled as baseline

---

## LOW Severity Issues

### L1. Gallery JavaScript broken — `selectGalleryScene()` references DOM IDs not present; panels don't render.
### L2. TSR (2003) consistently last — should be labeled "Baseline" if used as sanity check.
### L3. Spec tables repeated 3× — should consolidate with tier columns.

---

*Revised by deep semantic analysis on 2026-03-03.*
