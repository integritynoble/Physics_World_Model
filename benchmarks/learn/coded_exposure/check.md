# Benchmark Review -- coded_exposure (Coded Exposure / Flutter Shutter)

**Review date:** 2026-03-03
**Reviewer:** Claude Opus 4.6 (automated)
**Benchmark URL:** https://pwm.platformai.org/benchmark/coded_exposure

---

## 1. Physics & Forward Model

Coded Exposure, also known as Flutter Shutter, is a computational photography
technique that temporally modulates the camera shutter during a single exposure
to encode scene information -- particularly motion blur -- in a way that is
more amenable to computational reconstruction than a conventional open shutter.

**Canonical DAG:** M --> C --> D (Modulation --> Convolution --> Detector)

| Stage | Physical process | Mathematical description |
|-------|-----------------|--------------------------|
| **M** (Modulation) | Spatio-temporal amplitude modulation via a coded shutter sequence | Binary or multi-level temporal code c(t) gates photon flux |
| **C** (Convolution) | Shift-invariant PSF convolution encoding motion blur | h(x) = integral of c(t) * delta(x - v*t) dt |
| **D** (Detector) | CCD/CMOS sensor readout with gain and noise | y = gain * (Poisson(signal) + Gaussian(read_noise)) |

**Signal equation:** y = PSF (convolution) x + noise

The key insight of coded exposure (Raskar et al., SIGGRAPH 2006) is that the
binary shutter code broadens the optical transfer function (OTF) so its
frequency spectrum avoids the deep nulls present in standard uniform-exposure
motion blur PSFs. This makes Wiener deconvolution significantly more stable.

**Carrier:** Photon (visible, 400-700 nm)
**Forward model type:** linear_operator (y = Ax + n, superposition holds)
**Category module:** compressive_mask
**Physics parameters:** sigma=2.0, read_noise=5.0 e-, pixel_size=6.5 um

---

## 2. Mismatch Parameters & Benchmark Structure

The PWM benchmark tests algorithm robustness under physics model mismatch --
the discrepancy between the true acquisition physics and what the
reconstruction algorithm assumes. Three mismatch parameters are defined:

| Parameter | Nominal | Perturbed range | Unit | Physical meaning |
|-----------|---------|-----------------|------|------------------|
| Shutter code timing error (s_c) | 0.0 | [-5.0, 5.0] | - | Jitter/drift in shutter open/close transitions |
| Motion blur PSF mismatch (m_b) | 0.0 | [0.0, 20.0] | velocity error | Incorrect assumed velocity in motion kernel |
| Sensor readout noise (s_r) | 5.0 | [1.0, 15.0] | e- | Electronics noise floor varies with temperature/gain |

**Three-tier evaluation structure:**

| Tier | Mismatch severity | Purpose | Ground truth | Scenes |
|------|-------------------|---------|--------------|--------|
| Public | Mild | Algorithm development, debugging | Provided | 5 |
| Dev | Moderate | Validation, hyperparameter tuning | Blind (server-scored) | 5 |
| Hidden | Severe | Final leaderboard ranking | Blind (containerized submission) | 5 |

The gap between public-tier and hidden-tier performance reveals an algorithm's
true robustness to model errors. The hidden tier requires Dockerized submission.

**Mismatch levels (maturity ladder):**
- M0: No mismatch (perfect forward model)
- M1: Single parameter perturbed
- M2: 3+ parameters simultaneously perturbed (compound)
- M3: Real calibration/experimental errors
- M4: Adversarial worst-case mismatch optimized to maximize failure

**Signal dimensions:** x = [64, 64], y = [64, 64]
**Expanded configs offer:** 128x128, 256x256, 512x512 image sizes
**Noise levels:** Clean (60 dB), Low (40 dB), Medium (30 dB), High (20 dB)
**Total benchmark cases:** 192 (B1: 12, B2: 60, B3: 60, B4: 60)

---

## 3. Reconstruction Methods & Leaderboard

### Registered solvers in PWM

| Tier | Solver | Module | GPU | Params |
|------|--------|--------|-----|--------|
| traditional_cpu | Adjoint | pwm_core.recon.adjoint | No | 0 |
| best_quality | PnP-ADMM | pwm_core.recon.pnp_admm | Yes | 2M |

**Default solver:** wiener_deblur

### Online leaderboard (from pwm.platformai.org)

| Rank | Method | Overall score | Notes |
|------|--------|---------------|-------|
| 1 | Uformer + gradient | 0.756 | Wang et al., CVPR 2022; transformer-based |
| 2 | HDR-CNN + gradient | 0.669 | Eilertsen et al., ACM TOG 2017 |
| 3 | PnP-FFDNet + gradient | 0.655 | Zhang et al., 2017; plug-and-play prior |
| 4 | Restormer + gradient | -- | Listed in QA check but no posted score |

**Composite score formula:**
0.4 x PSNR_norm + 0.4 x SSIM + 0.2 x (1 - ||y - H_hat x_hat|| / ||y||)

The 20% consistency term penalizes reconstructions that do not agree with the
(corrected) forward model, encouraging physically plausible solutions rather
than hallucinated textures.

**Primary metric:** PSNR
**Secondary metric:** SSIM

### Four benchmark tasks

| Task | Description |
|------|-------------|
| B1 (Design) | Spec DAG: design shutter code, exposure time, motion range |
| B2 (Forward + Reconstruct) | Reconstruct under mismatch: timing error + unknown motion |
| B3 (System Identification) | Estimate true shutter function and motion kernel from data |
| B4 (Correct + Diagnose) | Correct shutter timing and motion model; target rho >= 0.80 |

---

## 4. Literature & State of the Art (2024-2025)

Recent advances in coded exposure and motion deblurring are driven by
transformer architectures, event cameras, and learned sensor designs:

**Coded exposure + neural representations (2024):**
Lightweight High-Speed Photography Built on Coded Exposure and Implicit Neural
Representation of Videos (IJCV 2024) combines classical flutter-shutter
acquisition with implicit neural representations to decode high-frame-rate
video from a single coded snapshot. This moves beyond single-frame deblurring
to temporal super-resolution.

**Adaptive deblurring with meta-learning (2025):**
Adaptive Image Deblurring CNNs with Meta-Tuning (Sensors 2025) address the
limitation of fixed-kernel CNN architectures by using meta-learning to adapt
deblurring networks to different blur kernels at test time -- directly relevant
to the PWM mismatch challenge where the true PSF is unknown.

**Event-based deblurring (NTIRE 2025):**
The NTIRE 2025 Challenge on Event-Based Image Deblurring introduced dual-encoder
architectures that separately process RGB frames and voxelized event streams,
fusing them via channel concatenation and transformer blocks. KANLinear layers
based on spline-interpolated kernels improve attention expressiveness. Event
cameras provide asynchronous, high-temporal-resolution data that complements
coded exposure by resolving motion ambiguity.

**Learned spatially varying pixel exposures (ICCP 2022, continued 2024):**
L-SVPE frameworks (Computational Imaging Lab, Stanford) combine next-generation
focal-plane sensor-processors with learned per-pixel exposure codes and ML-based
deblurring. This generalizes coded exposure from temporal-only to spatio-temporal
modulation.

**AIM 2025 Challenge on High FPS Motion Deblurring (ICCV 2025W):**
Efficient real-world deblurring methods targeting high frame-rate reconstruction
from motion-blurred inputs, with emphasis on practical deployment constraints.

**Compressive imaging beyond sensor resolution (Optics & Lasers Eng., 2023):**
Coded exposure combined with time-delay integration (TDI) achieves
super-resolution beyond the physical sensor pixel count, extending the coded
exposure paradigm to remote sensing applications.

**Key takeaway for PWM benchmark:** The top leaderboard entry (Uformer) uses a
CVPR 2022 architecture. Newer transformer variants (Restormer, NAFNet),
event-camera fusion, and meta-learned adaptation offer concrete paths to
improvement. The blind mismatch setting of PWM is particularly well-suited
to meta-learning and plug-and-play approaches that can adapt at test time.

---

## 5. Local Dataset Status

**Local dataset directory:** datasets/benchmark/coded_exposure/ -- DOES NOT EXIST

The coded_exposure modality does not have a local dataset directory under
datasets/benchmark/. The other active benchmark modalities (cacti, cbct,
cryo_em, ct, mri, sd_cassi, spc_kronecker, ultrasound) all have local
dataset directories populated.

**Data source configuration (from coded_exposure.yaml):**
- Primary: HDR+ Burst Dataset (https://hdrplusdata.org/), Hasinoff et al.,
  SIGGRAPH Asia 2016, Research use license
- Fallback: Synthetic generation via shepp_logan phantom
- Expanded config also lists: coded_exposure_generated (synthetic)

**Action needed:** The coded_exposure dataset needs to be downloaded or
generated before local benchmarking can proceed. The benchmark runner at
benchmarks/runners/run_expanded.py --modality coded_exposure should handle
fallback to the synthetic generator, but real-data evaluation requires
downloading from the HDR+ dataset.

**Learning materials:** All 5 curriculum files are present and correctly sized:
- README.md (1,468 B)
- 01_physics_fundamentals.md (2,235 B)
- 02_forward_model.md (2,721 B)
- 03_reconstruction_algorithms.md (2,042 B)
- 04_pwm_benchmark.md (2,449 B)
- 05_hands_on_tutorial.md (3,548 B)

**QA check status (from scripts/check_modality.py v2):** PASS -- 23/23 checks
passed, 0 errors, 0 warnings, 2 info items.

---

## 6. Comprehensive Assessment

### Strengths

1. **Well-defined physics model.** The M-->C-->D DAG cleanly separates
   modulation, convolution, and detection. The linear forward model enables
   a wide range of classical and learned solvers.

2. **Meaningful mismatch parameters.** The three mismatch axes (timing jitter,
   velocity error, readout noise) correspond to real-world failure modes in
   flutter-shutter cameras. The severity ladder from M0 to M4 provides a
   principled difficulty progression.

3. **Strong infrastructure.** Benchmark config, expanded config, QA checks,
   5-part learning curriculum, and online leaderboard with 4 submitted methods
   are all operational. The 192 total benchmark cases across 4 tasks (B1-B4)
   provide comprehensive coverage.

4. **Composite scoring is physics-aware.** The 20% consistency term
   (measurement fidelity) discourages adversarial or hallucinatory
   reconstructions that score well on PSNR/SSIM but violate the forward model.

### Weaknesses & Gaps

1. **No local dataset.** The datasets/benchmark/coded_exposure/ directory
   does not exist. This is the most critical gap -- other modalities (ct, mri,
   ultrasound, etc.) already have populated dataset directories. Without local
   data, the benchmark cannot be run offline.

2. **Small image size.** The default config uses 64x64 images. While the
   expanded config supports up to 512x512, modern deblurring methods are
   typically evaluated at 256x256 or higher. The small default may not
   adequately test the spatial frequency recovery that distinguishes coded
   exposure from conventional deblurring.

3. **Narrow solver roster.** Only two solvers are registered (Adjoint and
   PnP-ADMM). The leaderboard shows Uformer and HDR-CNN performing well,
   but these are not in the local solver registry. Adding Restormer, NAFNet,
   or Uformer to solver_registry.yaml would strengthen reproducibility.

4. **Missing reference PSNR.** The config has reference_psnr: null and
   expected_psnr_range: null. Establishing expected performance bounds would
   help users calibrate their results and detect implementation errors.

5. **Maturity M0.** The modality is at the lowest maturity level, meaning only
   template-level forward models are validated. Advancing to M1 (synthetic
   validation) or M2 (compound mismatch) would increase benchmark credibility.

6. **Leaderboard is thin.** Only 4 methods with scores, and the gap between
   rank 1 (0.756) and rank 3 (0.655) is large. Recruiting more submissions --
   especially from the 2024-2025 methods reviewed above -- would make the
   benchmark more informative.

### Recommended Actions (Priority Order)

| Priority | Action | Effort |
|----------|--------|--------|
| P0 | Create datasets/benchmark/coded_exposure/ with dev/hidden tiers from HDR+ data | Medium |
| P1 | Set reference_psnr and expected_psnr_range in config | Low |
| P1 | Register Uformer and Restormer in solver_registry.yaml | Low |
| P2 | Increase default image size to 256x256 to match modern standards | Low |
| P2 | Advance maturity from M0 to M1 with synthetic data validation | Medium |
| P3 | Add event-camera fusion baseline (NTIRE 2025 style) | High |
| P3 | Implement meta-learning adapter for blind PSF mismatch | High |

### Overall Verdict

The coded_exposure benchmark has solid foundations: correct physics, meaningful
mismatch parameters, functional infrastructure, and a composite metric that
enforces physical consistency. The primary gap is the missing local dataset,
which blocks offline evaluation. The modality is ready for M0-->M1 maturity
advancement once data is provisioned and reference performance is established.

---

**Tags:** coded_exposure, flutter_shutter, computational_photography,
motion_deblur, blind_deconvolution, physics_mismatch, linear_inverse_problem,
wiener_deblur, PnP-ADMM, Uformer, HDR_plus, benchmark_review