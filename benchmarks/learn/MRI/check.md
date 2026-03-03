# Benchmark QA Check — MRI

**URL:** https://pwm.platformai.org/benchmark/mri
**HTTP Status:** 200
**Check Date:** 2026-03-03 (deep semantic review)

## Summary

| Severity | Count |
|----------|-------|
| HIGH | 8 |
| MEDIUM | 12 |
| LOW | 10 |

---

## HIGH Severity Issues

### H1. Negative values for inherently positive physical quantities
The spec ranges contain negative values for quantities that represent error magnitudes:
- `gradient_nonlin`: -2.0 to 4.0 % (Public), -1.4 to 4.6 % (Hidden) — gradient nonlinearity is always non-negative
- `coil_sensitivity`: -5.0 to 10.0 % (Public), -3.5 to 11.5 % (Hidden) — amplitude error should be positive
- `k_trajectory`: -1.0 to 2.0 % (Public), -0.7 to 2.3 % (Hidden) — trajectory deviation magnitude is always positive
**Fix:** Either clarify these are *signed deviations* (not magnitudes), or clamp to [0, max].

### H2. PSNR_norm undefined in scoring formula
Scoring formula: `0.4 × PSNR_norm + 0.4 × SSIM + 0.2 × (1 − ‖y − Ĥx̂‖/‖y‖)`
- "PSNR_norm" normalization method not specified (min-max? reference baseline? dynamic range?)
- Norm type (L2? Frobenius?) in data-fidelity term not stated
- Not clear if PSNR is computed in magnitude or complex domain
**Fix:** Define PSNR_norm = (PSNR - PSNR_min) / (PSNR_max - PSNR_min) with bounds stated, and specify L2 norm.

### H3. Spec ranges don't monotonically increase in difficulty
Expected: Public (easiest) → Dev → Hidden (hardest), but ranges don't nest:
- B0_inhomog max: 3.0 (Public) → 2.7 (Dev) → 3.45 (Hidden) — *Dev is easier than Public on the max side*
- gradient_nonlin min: -2.0 (Public) → -2.4 (Dev) → -1.4 (Hidden) — *Hidden narrower than Dev*
**Fix:** Ensure ranges strictly widen: Public ⊂ Dev ⊂ Hidden.

### H4. HDF5 submission format undocumented
Page says "submit Reconstructed signals and corrected spec as HDF5" but specifies:
- No HDF5 key/group structure (e.g., `/kspace`, `/image`, `/spec`)
- No data types (complex64 vs float32)
- No array shapes (e.g., [num_coils, height, width])
- No metadata requirements
**Fix:** Add a "Submission Format" section with exact HDF5 schema.

### H5. Leaderboard rank inversion on Hidden tier unexplained
- Public/Dev: CS-Wavelet ranks 3rd, MoDL ranks 2nd
- Hidden: CS-Wavelet jumps to 2nd (0.681), MoDL drops to 3rd (0.652)
- No discussion of why rankings invert, undermining Hidden tier credibility
**Fix:** Add note explaining rank changes (e.g., "CS-Wavelet is more robust to severe mismatch").

### H6. Forward model notation inconsistent between sections
- Early: `y_c = F_u * S_c * x + n_c` (multi-coil parallel imaging model)
- Later: `y = M * FFT2(x)` (single-channel undersampled model)
- These represent fundamentally different physical models
**Fix:** Use one consistent equation; if both are needed, explicitly show the relationship.

### H7. Submission format contradicts between tiers
- Public/Dev: "submit Reconstructed signals and corrected spec as HDF5"
- Hidden: "Submit Docker container / Python script accepting y + H, outputting x_hat + corrected spec"
- Two incompatible submission mechanisms with no explanation of which is canonical
**Fix:** Clarify the two-track submission or unify to Docker for all tiers.

### H8. Forward operator H delivery format unspecified
States input includes "ideal forward operator (H)" but doesn't say whether H is:
- Dense matrix (infeasible for 320×320 images at ~10GB per matrix)
- Functional operator (Python callable)
- Implicit (undersampled FFT defined by mask + coil maps)
**Fix:** Specify that H is defined implicitly via mask, coil_maps, and FFT.

---

## MEDIUM Severity Issues

### M1. Noise model incomplete
States "complex Gaussian noise" but provides no SNR, σ, or noise power specification across tiers.

### M2. Spec primitives declared but not fully mapped
Page lists 11 primitives (P, M, Π, F, C, Σ, D, S, W, R, Λ) but Spec DAG shows only `F(k-traj) → D(g, η₁)`. Missing: how B₀, ΔG, ΔS map to these primitives.

### M3. Larmor frequency not stated
3T field strength stated but Larmor frequency (~127.7 MHz for ¹H) never mentioned, relevant for B₀ ppm context.

### M4. Acceleration factor inconsistency
States "Acceleration Factor: 4" but also "Center Fraction: 0.08" with variable-density random sampling, implying variable effective acceleration.

### M5. Missing key references
- GRAPPA (Griswold et al. MRM 2002) — used in leaderboard, not in references
- CS-MRI (Lustig et al. MRM 2007) — used in leaderboard, not in references
- MoDL (Aggarwal et al. IEEE TMI 2019) — leaderboard method, not referenced
- SwinMR (Huang et al. MICCAI 2022) — top performer, not referenced

### M6. Existing references lack DOIs
- Pruessmann et al. (1999): no DOI
- Zbontar et al. (2018): arXiv only, no journal version
- Sriram et al. (2020): missing full title and DOI

### M7. Gallery Scenario IV unlabeled
Comparison section at bottom appears to be Scenario IV (Blind Calibration) but is not labeled as such. Only Scenarios I, II, III are explicitly shown in gallery.

### M8. Gallery has no per-algorithm PSNR/SSIM
Algorithm comparison section shows 12 images (4 scenes × 3 algorithms) with zero quantitative metrics. Contradicts earlier gallery that shows metrics.

### M9. Scene naming convention unclear
Table uses "challenge_sample_0/1/2, aug_3" — the "aug_" prefix is not explained (augmented? augmentation?).

### M10. Multi-coil combination not explained in forward model
Describes per-coil equation `y_c = F_u * S_c * x + n_c` but never explains how individual coil data combines for reconstruction (RSS? SENSE? Adaptive combine?).

### M11. Sample count ambiguity
States "Public 3 scenes, Dev 3 scenes, Hidden 3 scenes" = 9 total, but relationship to fastMRI dataset (1594 knee + 6970 brain volumes) unclear.

### M12. Degradation numbers inconsistent
States "Degradation: -0.1 dB" and "Recovery: ±3.0 dB" but doesn't define which scenario pair, and "±" is nonsensical for a directional metric.

---

## LOW Severity Issues

### L1. Placeholder action links
`/benchmark/mri/compete` and `/benchmark/mri/contribute` — may not resolve.

### L2. Section anchor links unvalidated
`/benchmark/mri/challenge/dev#submission-area` — anchor may not exist on page.

### L3. Duplicate navbar links
"Physics World Model" and "Benchmarks" both link to `/benchmark`.

### L4. No alt-text on gallery images
12+ images in gallery lack figure captions and alt-text for accessibility.

### L5. GCS image paths use two CDN structures
Gallery: `/gcs/img/benchmark_gallery/mri/...` vs Setup: `/static/img/setups/mri.png` — potential CDN mismatch.

### L6. Spec DAG diagram has no figure caption.

### L7. 15 receive coils stated but not connected to coil sensitivity error parameter.

### L8. Operator "*" ambiguous (composition vs convolution) in forward model.

### L9. SSIM window size and data range not specified (default 7×7? data_range=1.0?).

### L10. The "aug_3" scene name suggests data augmentation but benchmark should use real samples only.

---

*Revised by deep semantic analysis on 2026-03-03. Original auto-check preserved in git history.*
