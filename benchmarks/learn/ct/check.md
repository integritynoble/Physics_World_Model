# Benchmark QA Check — CT

**URL:** https://pwm.platformai.org/benchmark/ct
**HTTP Status:** 200
**Check Date:** 2026-03-03 (deep semantic review)

## Summary

| Severity | Count |
|----------|-------|
| HIGH | 8 |
| MEDIUM | 12 |
| LOW | 7 |

---

## HIGH Severity Issues

### H1. Leaderboard rank inversion unexplained
- Public: DuDoTrans #1 (0.838), DOLCE #2 (0.823)
- Dev: DOLCE #1, DuDoTrans #2 — rankings reverse
- Hidden: DuDoTrans drops to #4 (0.671) despite leading Public
- DuDoTrans degrades -0.167 vs DOLCE -0.104 — possible overfitting not discussed
**Fix:** Add note explaining rank changes across tiers.

### H2. Composite scoring formula unverifiable
Formula: `0.4 × PSNR_norm + 0.4 × SSIM + 0.2 × (1 − ‖y − Ĥx̂‖/‖y‖)`
- PSNR_norm normalization not defined (min-max? reference baseline?)
- Consistency term uses Ĥ (corrected operator) — but ground-truth H not disclosed for Dev/Hidden
- Cannot reproduce composite scores from tier-specific values
**Fix:** Add worked example showing composite score calculation for one method.

### H3. Sinogram shape contradiction: 60 views vs 180 angles
- Experimental setup: "Num Views: 60"
- Gallery description: sinogram shape "(180, 64)"
- These are 3× different — which is correct?
**Fix:** Reconcile and state the actual sinogram dimensions clearly.

### H4. Forward model operator notation conflated
- Equation section: `y = R*x + n` (R = Radon transform)
- Spec primitives: uses Π symbol for projection
- Consistency metric: uses Ĥ for forward operator
- Three different symbols (R, Π, Ĥ) for the same operator with no mapping stated
**Fix:** Define operator naming convention once and use consistently.

### H5. Beam hardening spec range: negative β is unphysical
- `beam_hardening_beta`: Public [-0.1, 0.2], Dev [-0.07, 0.23], Hidden [0.07, 0.37]
- Negative beam hardening coefficient is physically unphysical (always over-corrects to under-correct)
- Hidden tier entirely positive while Public/Dev allow negative — no justification
**Fix:** Document what negative β means or clamp to [0, max].

### H6. Dataset shapes and dtypes undocumented
- sinogram_measured shape: not stated (n_angles × n_detectors = ?)
- x_true shape: not stated (e.g., 512×512)
- Data type (float32/float64) not specified
- Value range (HU units? attenuation coefficients? [0,1]?) not stated
**Fix:** Add explicit shape/dtype/range documentation per HDF5 key.

### H7. Oracle scenario math inconsistent
- Ideal (I): 37.7 dB, Mismatch (II): 36.39 dB → Degradation: -1.31 dB
- Oracle (III): 35.12 dB — *worse than Mismatch*, not a recovery
- Claims "Recovery: ±1.3 dB" which contradicts III < II
**Fix:** Clarify what Oracle achieves and fix the degradation/recovery narrative.

### H8. Angle error range asymmetry unjustified
- `angle_error_deg`: Public [-6.5, 9.5], Dev [-5.0, 11.0], Hidden [-2.0, 14.0]
- Negative range shrinks while positive expands — why asymmetric?
- No physical justification for biased error distribution
**Fix:** Justify asymmetry or make ranges symmetric.

---

## MEDIUM Severity Issues

### M1. Beam hardening physics model missing from forward model
Only correction methods listed (polynomial, dual-energy), but the *forward* beam hardening model (how it corrupts data) is absent.

### M2. Metal artifact disconnected from forward model
Hidden tier mentions "adversarial modifications (metal inserts)" but no metal artifact physics in the forward model section.

### M3. Noise model unspecified
Equation includes `n` but never defines distribution (Poisson? Gaussian? compound?), σ, or SNR level.

### M4. Spec range progression not monotonic
Ranges don't consistently widen from Public→Dev→Hidden. `center_offset_px` negative range: -4→-3→-1 (shrinking).

### M5. Algorithm comparison section has no quantitative metrics
Shows 6 algorithms × 4 scenes visually but provides zero PSNR/SSIM numbers.

### M6. Dataset citation inconsistency
LoDoPaB-CT cited as "arXiv 2019" in one place and "Scientific Data 8, 109 (2021)" elsewhere — two different citations for same paper.

### M7. Missing key references and DOIs
- AAPM Grand Challenge (McCollough et al. 2017): attributed in header but not in references
- Feldkamp (1984) cone-beam reconstruction: not cited
- No DOIs on any references

### M8. Inconsistent algorithm name capitalization
Gallery: "Dolce", "Fbp", "Fbpconv" vs Leaderboard: "DOLCE", "FBP", "FBPConvNet".

### M9. Radon transform notation non-standard
Uses `s` for line offset (standard is `ρ` or `t`). Minor but confusing for CT practitioners.

### M10. Scenario IV defined in header but never explained
Section title mentions "Scenario IV: Blind Calibration" but no methodology description provided.

### M11. Patient-to-scene mapping unclear
Dev tier references "validation split, patients 0-63" (64 patients) for 20 scenes — unclear mapping.

### M12. Sampling theorem not validated
60 views for 512×512 reconstruction — no mention of Nyquist criterion or why this undersampling is justified.

---

## LOW Severity Issues

### L1. Download URLs not directly visible in page content.
### L2. Submission endpoint/format for Docker containers not specified.
### L3. GCS image paths unverified for rendering.
### L4. JavaScript gallery scene selector HTML incomplete in content.
### L5. No image alt-text for accessibility.
### L6. Spec DAG diagram has no figure caption.
### L7. "Common Mistakes" section could link to specific solutions.

---

*Revised by deep semantic analysis on 2026-03-03.*
