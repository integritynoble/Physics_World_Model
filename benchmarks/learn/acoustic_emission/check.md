# Benchmark QA Check — acoustic_emission

**URL:** https://pwm.platformai.org/benchmark/acoustic_emission
**HTTP Status:** 200
**Check Date:** 2026-03-03 (deep semantic review)

## Summary

| Severity | Count |
|----------|-------|
| HIGH | 3 |
| MEDIUM | 12 |
| LOW | 3 |

---

## HIGH Severity Issues

### H1. Leaderboard Challenge vs Public score mismatch
- Challenge leaderboard ranks SwinIR first (0.687) overall
- Public leaderboard ranks WaveNet first (0.778)
- Overall score (0.687) differs from Public score (0.773) with no explanation of how composite is calculated
**Fix:** Show worked example of composite scoring; clarify if Challenge = weighted average across tiers.

### H2. Gallery section missing / not rendering
- JavaScript function `selectGalleryScene()` and `gallery-scene-panel` exist in HTML
- Panels hidden by default with no visible toggle — no actual gallery images render
**Fix:** Fix gallery panel visibility or include static fallback images.

### H3. HDF5 dataset schema undocumented
- Mentions "Load HDF5" but provides zero schema information
- Missing: key names, data types, array dimensions (time × channels × scenes)
- x_true format undefined — is it pressure field? dipole moment tensor?
- No sampling rate, number of sensors, number of time samples, scene geometry
**Fix:** Document full HDF5 schema with example loading code.

---

## MEDIUM Severity Issues

### M1. Forward model has no equations
DAG shows "P → S → D" but no mathematical formalization. No explicit y = Hx + η equation. Mismatch parameters not formally connected to forward operator.

### M2. Propagation kernel unspecified
States "Fresnel, Rayleigh-Sommerfeld" as examples but doesn't clarify which is used. Fresnel assumes paraxial (invalid for broad-angle wave propagation).

### M3. No dispersion model
Real AE signals exhibit frequency-dependent attenuation and dispersion in solid media — not mentioned.

### M4. Spec range: wave speed ranges overlap inconsistently across tiers
Public (5860–5980), Dev (5852–5972), Hidden (5872–5992) m/s — ranges don't nest and progression rationale missing.

### M5. Spec range: source location error too small
±2.3 mm (Hidden) is small for AE sensor networks; typical positioning errors are centimeters. No justification.

### M6. Spec range: sensor coupling gain unrealistically tight
0.96–1.08 (4–8% variation) is modest; real coupling losses can exceed 20 dB.

### M7. PSNR_norm undefined in scoring formula
Same issue as other modalities — normalization method, range, clipping rules not specified.

### M8. Consistency metric ignores noise floor
`1 − ‖y − Ĥx̂‖/‖y‖` assumes residual → 0 for perfect reconstruction, but measurement noise η makes this non-zero by design.

### M9. No detector noise model specified
Lists "Gaussian, Poisson, mixed" as options but doesn't state which is used. Real AE sensors exhibit 1/f noise.

### M10. Incomplete references
- Missing DOIs for all citations
- No AE domain references (Vallen, Grosse on AE testing)
- No benchmark design methodology citation
- WaveNet citation incomplete (missing authors)

### M11. Download buttons ambiguous
"Download" appears twice per tier without specifying content (data only? code template?).

### M12. No Docker template or baseline code provided
"Submit Algorithm" promises Docker support but no template, example container, or API spec.

---

## LOW Severity Issues

### L1. Spec tables repeated 3× — should consolidate into one table with tier columns.
### L2. DAG "P → S → D" is text-only — needs a schematic diagram.
### L3. Math notation inconsistent (inline norms vs subscript notation).

---

*Revised by deep semantic analysis on 2026-03-03.*
