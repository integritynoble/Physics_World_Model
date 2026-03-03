# Comprehensive Benchmark QA Check — Adaptive Optics

**URL:** <https://pwm.platformai.org/benchmark/adaptive_optics>
**HTTP Status:** 200
**Check Date:** 2026-03-03 (comprehensive 6-point review)

---

## 1. Benchmark Page Errors

### Severity Summary

| Severity | Count |
|----------|-------|
| HIGH     | 4     |
| MEDIUM   | 7     |
| LOW      | 3     |

---

### HIGH-1. Dataset source attribution is wrong — SEG/EAGE Salt Model (geophysics) cited for AO

The benchmark page attributes the dataset to "Aminzadeh et al., SEG 1997" (the
SEG/EAGE Salt Model), which is a geophysics/seismic dataset. Adaptive optics
benchmarks should reference astronomical, ophthalmological, or microscopy
datasets. The local config (`adaptive_optics.yaml`) confirms `dataset_id: ''`,
`citation: ''`, and `fallback: generated` with `synthetic_generator: shepp_logan`.
The Shepp-Logan phantom is a medical CT phantom — also unrelated to AO. This
appears to be a copy-paste error from another modality.

**Severity:** HIGH
**Fix:** Replace dataset source with an AO-relevant dataset (e.g., astronomical
point sources degraded by turbulence, retinal images from AO-ophthalmoscopy, or
simulated Kolmogorov phase screens). Remove the SEG/EAGE citation.

---

### HIGH-2. PSNR_norm undefined in the composite scoring formula

The scoring formula is stated as:
```
0.4 * PSNR_norm + 0.4 * SSIM + 0.2 * (1 - ||y - H_hat * x_hat|| / ||y||)
```
The normalization method for `PSNR_norm` is not specified anywhere on the page.
Without knowing the mapping from raw PSNR (dB) to a [0, 1] score, the composite
metric is not reproducible. Additionally, the local config (`04_pwm_benchmark.md`)
lists `psnr` as the primary metric and `ssim` as secondary, with no mention of
the consistency term or composite formula.

**Severity:** HIGH
**Fix:** Define PSNR_norm explicitly, e.g., `PSNR_norm = clip((PSNR - 20) / 20, 0, 1)`,
and publish the normalization bounds. Reconcile the page formula with the local
config which only references raw PSNR as primary.

---

### HIGH-3. Leaderboard tier-aggregation weights are not disclosed

The overall composite scores cannot be reproduced from tier-level scores. For
example:
- AO-ViT: Public=0.768, Dev=0.718, Hidden=0.646, Overall=0.711
- WFNet: Public=0.707, Dev=0.638, Hidden=0.582, Overall=0.642
- Zernike LS: Public=0.631, Dev=0.618, Hidden=0.600, Overall=0.616
- PnP-ADMM: Public=0.689, Dev=0.576, Hidden=0.567, Overall=0.611

Simple arithmetic mean of AO-ViT tiers = (0.768+0.718+0.646)/3 = 0.711 (matches).
WFNet: (0.707+0.638+0.582)/3 = 0.642 (matches). So the tiers appear to be
equally weighted (1/3 each), but this is not stated on the page. Users must be
able to verify scores.

**Severity:** HIGH
**Fix:** State explicitly: "Overall = (Public + Dev + Hidden) / 3" or whatever
the actual aggregation is. Show a worked example.

---

### HIGH-4. Forward model DAG omits AO-critical control-loop components

The DAG is `M --> C --> D` (Modulation, Convolution, Detector), which is a
generic PSF-convolution pipeline. For adaptive optics, the DAG should include:
- Wavefront sensor (WFS) measurement
- Deformable mirror (DM) correction loop
- Temporal servo dynamics (servo lag is a mismatch param but not in DAG)
- Atmospheric turbulence layer (Kolmogorov phase screen)

The local config confirms `category_module: microscopy_psf`, which is a
microscopy module being reused for AO. The signal equation `y = PSF * x + noise`
is correct but the PSF should be parameterized by the residual wavefront error
after AO correction, not treated as a generic convolution kernel.

**Severity:** HIGH
**Fix:** Extend the DAG to `Atmosphere --> WFS --> DM --> PSF(residual) --> Detector`
or at minimum annotate that the PSF encodes the residual AO correction error.
Use a dedicated AO physics engine instead of reusing `microscopy_psf`.

---

### MEDIUM-1. Mismatch parameter ranges on the page differ from the local config

**Page (extracted per-tier ranges):**
| Parameter          | Public         | Dev            | Hidden         |
|--------------------|----------------|----------------|----------------|
| dm_actuator_gain   | [0.98, 1.04]   | [0.976, 1.036] | [0.986, 1.046] |
| wfs_centroid_bias  | [-0.04, 0.08]  | [-0.048, 0.072]| [-0.028, 0.092]|
| fried_parameter_r0 | [0.13, 0.19]   | [0.126, 0.186] | [0.136, 0.196] |
| servo_lag          | [-0.4, 0.8]    | [-0.48, 0.72]  | [-0.28, 0.92]  |

**Local config (`adaptive_optics.yaml`):**
| Parameter          | Range       |
|--------------------|-------------|
| DM actuator gain   | [0.9, 1.1]  |
| WFS centroid bias  | [-0.2, 0.2] |
| Fried parameter r0 | [0.1, 0.25] |
| Servo lag          | [0.0, 2.0]  |

The local config has wider ranges (full envelope) while the page shows narrower
per-tier ranges. The page ranges are asymmetric around the nominal and the tiers
overlap heavily, without a clear monotonic difficulty progression (e.g., Hidden
r0 range [0.136, 0.196] is narrower and shifted up vs. Dev [0.126, 0.186]).

**Severity:** MEDIUM
**Fix:** Clarify that local config contains the outer envelope and the page shows
per-tier subsets. Make tier ranges monotonically harder (wider or more adverse).

---

### MEDIUM-2. Servo lag has negative values (physically nonsensical)

The page lists servo_lag ranges that include negative values (e.g., Public:
[-0.4, 0.8] ms). Servo lag is a temporal delay in the AO control loop and
cannot be negative in physical systems.

**Severity:** MEDIUM
**Fix:** Clamp servo_lag minimum to 0.0 ms, or rename the parameter to
"servo_timing_offset" if negative values represent predictive correction.

---

### MEDIUM-3. AO physics underspecified in mismatch parameters

- `wfs_centroid_bias` implies a Shack-Hartmann WFS but no subaperture geometry,
  lenslet pitch, or number of subapertures is specified.
- `dm_actuator_gain` lacks actuator count, influence function type, and stroke
  range.
- `fried_parameter_r0` is given as a mismatch parameter but is actually an
  atmospheric condition — it should be the error in the assumed r0, not r0 itself.

**Severity:** MEDIUM
**Fix:** Add WFS and DM specifications. Redefine r0 mismatch as
`delta_r0 = r0_true - r0_assumed`.

---

### MEDIUM-4. Only 4 algorithms on leaderboard — insufficient for credible benchmark

The leaderboard has only 4 methods:
1. AO-ViT + gradient
2. WFNet + gradient
3. Zernike LS + gradient
4. PnP-ADMM (PSF) + gradient

All use "+ gradient" suffix suggesting a common post-processing step. No
pure deep learning methods (end-to-end), no RL-based controllers, no
diffusion-based reconstructors, and no classical iterative methods (e.g.,
Richardson-Lucy, Wiener filter) are represented.

**Severity:** MEDIUM
**Fix:** Add at minimum: Wiener filter baseline, Richardson-Lucy, a
diffusion-prior method, and an RL-based controller (see Section 4).

---

### MEDIUM-5. Missing domain-specific references

The page cites only method papers (Nishizaki 2019, Noll 1976, Venkatakrishnan
2013). Missing essential AO references:
- Roddier (1999) — "Adaptive Optics in Astronomy"
- Hardy (1998) — "Adaptive Optics for Astronomical Telescopes"
- Tyson (2015) — "Principles of Adaptive Optics"
- Kolmogorov turbulence model
- Fried (1966) — original r0 parameter definition
- Greenwood (1977) — temporal bandwidth

**Severity:** MEDIUM
**Fix:** Add a "References" section with foundational AO textbooks and papers.

---

### MEDIUM-6. Image dimensions inconsistency: config vs. expanded config

- `adaptive_optics.yaml`: x_shape = [64, 64], y_shape = [64, 64]
- `adaptive_optics_expanded.yaml`: small = [128, 128], standard = [256, 256],
  large = [512, 512]
- Neither matches what appears on the page (64x64 from the base config)

**Severity:** MEDIUM
**Fix:** Reconcile the base config with the expanded config. State which
resolution is used for the leaderboard evaluation.

---

### MEDIUM-7. Gallery images present but incomplete

Gallery images exist for scenes 00-03 under
`platform/pwm_platform/static/img/benchmark_gallery/adaptive_optics/` with
gt.png, measurement_I/II.png, and recon_I/II/III.png. However:
- No scene_04 (5 scenes claimed but only 4 in gallery)
- No indication which reconstruction method maps to recon_I, recon_II, recon_III

**Severity:** MEDIUM
**Fix:** Add scene_04 gallery images. Label reconstructions with method names.

---

### LOW-1. Expanded config has empty mismatch_params list

`adaptive_optics_expanded.yaml` line 42: `mismatch_params: []` — the expanded
config does not carry forward the mismatch parameters from the base config.

**Severity:** LOW
**Fix:** Populate `mismatch_params` in expanded config or reference base config.

---

### LOW-2. Maturity label "M0" undocumented

The config states `maturity: M0` but the maturity scale (M0-M4) is only
partially described in the expanded config's mismatch_levels section, conflating
maturity with mismatch severity.

**Severity:** LOW
**Fix:** Document the maturity scale separately from mismatch levels.

---

### LOW-3. Download button URLs not explicitly shown on page

Users cannot verify if dataset endpoints are active without clicking. No
checksums or file sizes are provided.

**Severity:** LOW
**Fix:** Show download URLs, file sizes, and SHA-256 checksums.

---

## 2. Local Dataset Inspection

### Dataset Directory

```
datasets/benchmark/adaptive_optics/   --> DOES NOT EXIST
```

No local dataset directory was found at the expected path. The config confirms
`fallback: generated` with `synthetic_generator: shepp_logan`, meaning the
benchmark currently runs entirely on synthetically generated Shepp-Logan phantoms.

### Gallery Assets (exist)

```
platform/pwm_platform/static/img/benchmark_gallery/adaptive_optics/
  scene_00/  gt.png, measurement_I.png, measurement_II.png, recon_I.png, recon_II.png, recon_III.png
  scene_01/  gt.png, measurement_I.png, measurement_II.png, recon_I.png, recon_II.png, recon_III.png
  scene_02/  gt.png, measurement_I.png, measurement_II.png, recon_I.png, recon_II.png, recon_III.png
  scene_03/  gt.png, measurement_I.png, measurement_II.png, recon_I.png, recon_II.png, recon_III.png
```

4 scenes present in gallery; 5 expected (scenes 00-04 for 5 public-tier scenes).

### Config Files (exist)

```
benchmarks/configs/adaptive_optics.yaml              (base config)
benchmarks/expanded_configs/adaptive_optics_expanded.yaml  (expanded config)
```

### Learn Materials (exist)

```
benchmarks/learn/adaptive_optics/
  README.md
  01_physics_fundamentals.md
  02_forward_model.md
  03_reconstruction_algorithms.md
  04_pwm_benchmark.md
  05_hands_on_tutorial.md
```

### Key Observations

- **No real AO data** anywhere in the repository.
- Ground truth is generated, not from experimental or simulated AO systems.
- The Shepp-Logan phantom (a CT/MRI test image) is inappropriate for AO.
- Solver registry has only 2 solvers: Adjoint (CPU baseline) and PnP-ADMM.
- The learn materials correctly describe AO physics but the config and data
  pipeline do not actually implement AO-specific forward modeling.

---

## 3. Public Dataset Source Assessment

### Current State: No Real Dataset

The benchmark currently uses no real AO dataset. The `dataset_id` is empty,
the `dataset_url` is empty, and the fallback is a Shepp-Logan phantom generator.

### Recommended AO Datasets

| Dataset | Source | Type | Size | Suitability |
|---------|--------|------|------|-------------|
| SPHERE/IRDIS (ESO) | ESO Archive | Astronomical AO | ~1000 observations | HIGH — real AO-corrected images with known PSF |
| Keck AO Archive | Keck Observatory | Astronomical AO | ~5000 frames | HIGH — well-characterized AO system |
| AOSLO Retinal Images | Various labs | Ophthalmic AO | ~500 images | MEDIUM — real AO with different physics |
| AOBench Simulation | Simulated | Synthetic AO | Configurable | HIGH — controlled ground truth, Kolmogorov turbulence |
| GPI Exoplanet Survey | Gemini | Extreme AO | ~600 targets | MEDIUM — coronagraphic, specialized |

### Recommendation

For a benchmark that tests robustness to AO model mismatch, the ideal approach is:

1. **Simulated AO data** with Kolmogorov phase screens and a realistic AO
   control loop (WFS + DM + servo), where ground truth is perfectly known.
2. **Real AO data** from a well-characterized system (e.g., SPHERE/VLT) where
   the PSF can be estimated from WFS telemetry, providing approximate ground
   truth via deconvolution with the telemetry-reconstructed PSF.

The current Shepp-Logan fallback provides no information about AO algorithm
performance and should be replaced urgently.

---

## 4. Algorithm Coverage Assessment

### Currently on Leaderboard

| # | Algorithm | Type | Year | Notes |
|---|-----------|------|------|-------|
| 1 | AO-ViT + gradient | Transformer-based wavefront sensing | 2024 | Top performer; likely based on vision transformer architecture |
| 2 | WFNet + gradient | CNN wavefront estimation | 2019/2021 | Nishizaki/Vera et al.; specialized for image-based WFS |
| 3 | Zernike LS + gradient | Classical least-squares | 1976 | Noll's Zernike decomposition; strong baseline |
| 4 | PnP-ADMM (PSF) + gradient | Plug-and-Play optimization | 2013 | Venkatakrishnan et al.; generic inverse problem solver |

### Notable Missing Algorithms

| # | Algorithm | Type | Year | Why Include | Reference |
|---|-----------|------|------|-------------|-----------|
| 1 | Wiener Filter / MMSE | Classical deconvolution | 1949/standard | Essential baseline — optimal linear filter for known PSF+noise | Standard textbook |
| 2 | Richardson-Lucy (RL) | Iterative ML deconvolution | 1972/1974 | Most widely used astronomical deconvolution algorithm | Richardson (1972), Lucy (1974) |
| 3 | GCViT (Pyramid WFS) | Transformer for wavefront sensing | 2024 | Best-performing architecture for non-modulated pyramid WFS; Strehl 0.28-0.77 on optical bench | Weinberg et al., A&A 2024 |
| 4 | ConvNeXt (WFS) | Modern CNN for wavefront sensing | 2024 | Strong CNN baseline; compared against GCViT | Weinberg et al., A&A 2024 |
| 5 | PO4AO / RL Controller | Reinforcement learning AO control | 2022-2024 | Policy-gradient optimization for real-time AO control; 3-5x contrast improvement | Landman et al., A&A 2022; Pou et al., 2024 |
| 6 | AOViFT | 3D vision Fourier transformer | 2025 | Aberration sensing from microscopy images without guide star; Nature Methods | Nature Methods 2025 |
| 7 | SFE-Net | Space-frequency encoding network | 2024 | Estimates PSF from biological images; 18 Zernike modes | bioRxiv 2023 / Photonics Res. 2024 |
| 8 | Self-supervised PSF deconv | Self-supervised blind deconvolution | 2025 | No ground truth needed; real-time OCT deconvolution | AI Photonics 2025 |
| 9 | Deep-RL WFS-less AO | Deep reinforcement learning | 2021 | Eliminates wavefront sensor entirely; DDPG-based | Hu et al., Opt. Express 2021 |
| 10 | ResNet-SHWFS | Modified ResNet for Hartmann-Shack | 2024 | CNN for clinical visual optics WFS | Scientific Reports 2024 |

### Coverage Gap Analysis

- **Classical methods:** 1/3 present (Zernike LS only; missing Wiener, RL)
- **CNN-based:** 1/4 present (WFNet only; missing GCViT, ConvNeXt, ResNet-SHWFS)
- **Transformer-based:** 1/2 present (AO-ViT; missing AOViFT)
- **RL-based:** 0/2 present (missing PO4AO, Deep-RL WFS-less)
- **Optimization-based:** 1/2 present (PnP-ADMM; missing blind deconvolution)
- **Diffusion/generative:** 0/1 present (no diffusion-prior methods)

---

## 5. Improvement Suggestions

### Priority 1: Critical Fixes (blocking credibility)

1. **Replace the Shepp-Logan phantom** with AO-appropriate ground truth. Generate
   synthetic AO data using Kolmogorov phase screens, a Shack-Hartmann WFS model,
   and a deformable mirror model. At minimum, simulate the full AO control loop
   to produce residual-wavefront PSFs.

2. **Fix the DAG** to include AO-specific components: atmosphere, WFS, DM, servo.
   The current `M --> C --> D` is indistinguishable from a generic microscopy
   benchmark.

3. **Define PSNR_norm** with explicit bounds and publish the scoring formula
   with a worked numerical example.

4. **Remove the SEG/EAGE Salt Model citation** — it has no relation to AO.

### Priority 2: Strengthen the Benchmark (important for community adoption)

5. **Add classical baselines:** Wiener filter and Richardson-Lucy are expected
   in any imaging reconstruction benchmark. Their absence makes the leaderboard
   seem incomplete.

6. **Add state-of-the-art methods:** GCViT (2024) and PO4AO (2022-2024) are
   the most-cited recent AO methods and should be on the leaderboard.

7. **Reconcile image dimensions:** The base config says 64x64, the expanded
   config says 128/256/512. State which is used for evaluation.

8. **Fix mismatch parameter ranges:** Remove negative servo lag values. Ensure
   tiers have monotonically increasing difficulty (wider ranges or more adverse
   parameter values for harder tiers).

9. **Add domain references:** Include Roddier, Hardy, Tyson, Fried, Greenwood,
   and Kolmogorov theory references on the benchmark page.

### Priority 3: Polish (nice-to-have)

10. **Complete the gallery:** Add scene_04 and label which reconstruction method
    corresponds to recon_I, recon_II, recon_III.

11. **Publish dataset schema:** Document HDF5 key names, array shapes, dtypes,
    and noise model parameters.

12. **Add download checksums:** Provide SHA-256 hashes and file sizes for all
    downloadable datasets.

13. **Populate expanded config:** Fill in `mismatch_params` in the expanded
    YAML so the full benchmark matrix is self-contained.

---

## 6. Action Items

| # | Action | Priority | Owner | Status |
|---|--------|----------|-------|--------|
| 1 | Replace Shepp-Logan generator with AO-specific synthetic data (Kolmogorov phase screens + WFS + DM simulation) | P0 | Data team | TODO |
| 2 | Fix forward model DAG to include AO control loop components | P0 | Physics team | TODO |
| 3 | Define and publish PSNR_norm bounds in scoring formula | P0 | Metrics team | TODO |
| 4 | Remove incorrect SEG/EAGE Salt Model citation | P0 | Content team | TODO |
| 5 | Add Wiener filter and Richardson-Lucy baselines to leaderboard | P1 | Algorithms team | TODO |
| 6 | Implement GCViT (Weinberg et al. 2024) and PO4AO (Landman et al. 2022) | P1 | Algorithms team | TODO |
| 7 | Reconcile 64x64 vs 128/256/512 image dimension discrepancy | P1 | Config team | TODO |
| 8 | Fix mismatch ranges: remove negative servo lag, ensure tier monotonicity | P1 | Physics team | TODO |
| 9 | Add foundational AO references (Roddier, Hardy, Tyson, Fried, Greenwood) | P1 | Content team | TODO |
| 10 | Add scene_04 gallery images and label reconstruction methods | P2 | Content team | TODO |
| 11 | Document HDF5 dataset schema (keys, shapes, dtypes) | P2 | Data team | TODO |
| 12 | Populate mismatch_params in expanded config YAML | P2 | Config team | TODO |
| 13 | Add download checksums and file sizes to page | P2 | Platform team | TODO |
| 14 | Evaluate and add diffusion-prior and self-supervised methods | P2 | Algorithms team | TODO |

---

## Appendix: Key References

### Foundational AO Theory
- Fried, D.L. (1966). "Optical Resolution Through a Randomly Inhomogeneous Medium for Very Long and Very Short Exposures." *JOSA*, 56(10), 1372-1379.
- Noll, R.J. (1976). "Zernike polynomials and atmospheric turbulence." *JOSA*, 66(3), 207-211.
- Greenwood, D.P. (1977). "Bandwidth specification for adaptive optics systems." *JOSA*, 67(3), 390-393.
- Hardy, J.W. (1998). *Adaptive Optics for Astronomical Telescopes*. Oxford University Press.
- Roddier, F. (1999). *Adaptive Optics in Astronomy*. Cambridge University Press.
- Tyson, R.K. (2015). *Principles of Adaptive Optics*. 4th edition, CRC Press.

### Algorithms on the Leaderboard
- Noll (1976) — Zernike polynomial decomposition (Zernike LS baseline)
- Nishizaki, Y. et al. (2019). "Deep learning wavefront sensing." *Opt. Express*, 27(1), 240-251. (WFNet precursor)
- Vera, E. et al. (2021). "WFNet: deep learning wavefront estimation." *Opt. Express*. (WFNet)
- Venkatakrishnan, S.V. et al. (2013). "Plug-and-Play Priors for Model Based Reconstruction." *IEEE GlobalSIP*. (PnP-ADMM)

### Missing Key Algorithms (recommended additions)
- Richardson, W.H. (1972). "Bayesian-Based Iterative Method of Image Restoration." *JOSA*, 62(1), 55-59.
- Lucy, L.B. (1974). "An iterative technique for the rectification of observed distributions." *Astron. J.*, 79, 745.
- [Weinberg, N. et al. (2024). "Transformer neural networks for closed-loop adaptive optics using nonmodulated pyramid wavefront sensors." *A&A*, 687, A202.](https://www.aanda.org/articles/aa/full_html/2024/07/aa49118-23/aa49118-23.html)
- [Landman, R. et al. (2022). "Towards on-sky adaptive optics control using reinforcement learning." *A&A*, 663.](https://www.aanda.org/articles/aa/full_html/2022/08/aa43311-22/aa43311-22.html)
- [Pou, B. et al. (2024). "Integrating supervised and reinforcement learning for predictive control with an unmodulated pyramid wavefront sensor." *Opt. Express*.](https://arxiv.org/html/2405.13610v1)
- [Nature Methods (2025). "Fourier-based three-dimensional multistage transformer for aberration correction in multicellular specimens." (AOViFT)](https://www.nature.com/articles/s41592-025-02844-7)

### Recent Surveys and Reviews
- [Liu et al. (2025). "Research Progress on Atmospheric Turbulence Perception and Correction Based on Adaptive Optics and Deep Learning." *Adv. Photonics Res.*, 2400204.](https://advanced.onlinelibrary.wiley.com/doi/10.1002/adpr.202400204)
- [MDPI Aerospace (2025). "A Review of Wavefront Sensing and Control Based on Data-Driven Methods."](https://www.mdpi.com/2226-4310/12/5/399)
- [ResearchGate (2022). "Adaptive optics based on machine learning: a review."](https://www.researchgate.net/publication/358168665_Adaptive_optics_based_on_machine_learning_a_review)
- [Scientific Reports (2024). "Experimental wavefront sensing techniques based on deep learning models using a Hartmann-Shack sensor."](https://www.nature.com/articles/s41598-024-80615-8)

---

*Comprehensive 6-point review on 2026-03-03. Sources: [Weinberg et al. 2024 (GCViT)](https://www.aanda.org/articles/aa/full_html/2024/07/aa49118-23/aa49118-23.html), [Liu et al. 2025 (AO+DL survey)](https://advanced.onlinelibrary.wiley.com/doi/10.1002/adpr.202400204), [Pou et al. 2024 (RL+PyWFS)](https://arxiv.org/html/2405.13610v1), [AOViFT (Nature Methods 2025)](https://www.nature.com/articles/s41592-025-02844-7), [Scientific Reports 2024 (ResNet-SHWFS)](https://www.nature.com/articles/s41598-024-80615-8), [Landman et al. 2022 (PO4AO)](https://www.aanda.org/articles/aa/full_html/2022/08/aa43311-22/aa43311-22.html), [Self-supervised PSF deconv 2025](https://doi.org/10.3788/ai.2025.10026), [MDPI WFS review 2025](https://www.mdpi.com/2226-4310/12/5/399).*
