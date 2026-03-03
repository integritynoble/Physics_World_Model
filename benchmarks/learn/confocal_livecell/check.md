# Benchmark Review -- confocal_livecell (Live Cell Confocal Microscopy)

**URL:** <https://pwm.platformai.org/benchmark/confocal_livecell>
**Review date:** 2026-03-03
**Reviewer:** Claude Opus 4.6 (automated)
**Local dataset:** NOT PRESENT (`datasets/benchmark/confocal_livecell/` does not exist)

---

## 1. Platform Page Audit

| Check | Result |
|-------|--------|
| Page loads (HTTP 200) | PASS |
| Title | Confocal Live-Cell -- Physics World Model |
| Modality description | Laser-scanning confocal microscopy for live-cell imaging; pinhole rejects out-of-focus fluorescence |
| Canonical DAG | C (PSF_confocal) --> D (g, eta) |
| Forward model principle | Convolution with confocal PSF (product of excitation and detection PSFs), Poisson-Gaussian noise |
| Scoring formula | 0.4 x PSNR_norm + 0.4 x SSIM + 0.2 x (1 - residual norm) |
| Tier structure | Public (5 scenes, GT provided), Dev (5 scenes, blind), Hidden (5 scenes, containerized submission) |
| Image specs | 512 x 512 pixels, 80 nm pixel size, 200 temporal frames, HDF5 format |
| Reference dataset | BioSR (Qiao et al., Nat. Methods 2024), DeepBacs fluorescence (Spahn et al., Commun. Biol. 2022) |
| Leaderboard entries | 4 methods present |
| Gallery images | 24/24 load OK |
| Learning materials | 5 modules + README (all present locally) |
| HDF5 data on GCS | Public and Dev tiers confirmed accessible |

### Mismatch Parameters (from platform)

| Parameter | Symbol | Nominal | Unit | Public Range | Dev Range | Hidden Range |
|-----------|--------|---------|------|-------------|-----------|-------------|
| Pinhole diameter error | Delta_ph | 0 | um | [-5.0, 10.0] | [-6.0, 9.0] | [-3.5, 11.5] |
| Refractive index | Delta_n | 1.515 | -- | [1.51, 1.525] | [1.509, 1.524] | [1.5115, 1.5265] |
| Photobleaching rate error | alpha_b | 0 | % | [-5.0, 10.0] | [-6.0, 9.0] | [-3.5, 11.5] |

### Local Config Mismatch Parameters (from `benchmarks/configs/confocal_livecell.yaml`)

| Parameter | Nominal | Range | Unit |
|-----------|---------|-------|------|
| PSF sigma | 1.5 | [0.8, 3.0] | px |
| Drift rate | 0.1 | [0.0, 1.0] | px/frame |
| Bleaching rate | 0.01 | [0.0, 0.1] | per frame |
| Pinhole misalignment | 0.0 | [0.0, 0.5] | AU offset |

**NOTE:** The platform page lists 3 mismatch parameters (pinhole diameter, refractive index, photobleaching rate), while the local YAML config lists 4 different parameters (PSF sigma, drift rate, bleaching rate, pinhole misalignment). This discrepancy should be reconciled -- either the local config needs updating to match the platform, or the platform should reflect the richer 4-parameter local model.

---

## 2. Leaderboard & Baseline Analysis

### Current Leaderboard (All 3 Tiers Combined)

| Rank | Method | Overall | Public | Dev | Hidden |
|------|--------|---------|--------|-----|--------|
| 1 | Restormer + gradient | 0.730 | 0.817 | 0.736 | 0.638 |
| 2 | PnP-FISTA + gradient | 0.683 | 0.736 | 0.668 | 0.645 |
| 3 | CARE + gradient | 0.655 | 0.773 | 0.617 | 0.575 |
| 4 | Richardson-Lucy + gradient | 0.612 | 0.635 | 0.614 | 0.588 |

### Top Method Detail (Restormer + gradient)

| Tier | PSNR (dB) | SSIM |
|------|-----------|------|
| Public | 34.7 | 0.966 |
| Dev | 30.08 | 0.918 |
| Hidden | 25.05 | 0.804 |

### Key Observations

- The Public-to-Hidden drop for Restormer is substantial: -9.65 dB PSNR and -0.162 SSIM. This indicates significant sensitivity to forward-model mismatch at unseen parameter ranges.
- PnP-FISTA shows the most stable cross-tier performance (Public-Hidden gap: only 0.091 in overall score), suggesting model-based priors generalize better than pure learned approaches under mismatch.
- Richardson-Lucy, despite being the simplest method, shows the smallest Public-Hidden gap (0.047), confirming that physics-based inversion is inherently more robust to distribution shift.
- All methods append "+ gradient" indicating gradient-based spec correction is used across the board.

### Local PWM Report Baselines (from `pwm/reports/confocal_livecell.md`)

| Workflow | Solver | PSNR | SSIM | Notes |
|----------|--------|------|------|-------|
| W1 (simulate+reconstruct) | Richardson-Lucy (50 iter) | 25.09 dB | 0.9378 | 64x64 synthetic phantom |
| W2 (corrected operator) | Richardson-Lucy (50 iter) | 25.33 dB | 0.9410 | +1.60 dB from PSF correction |
| Casepack (low-dose drift) | PnP-HQS | 26.27 dB | -- | 400 photon budget |

---

## 3. Physics & Forward Model Assessment

### Optical Setup

- **Instrument reference:** Zeiss LSM 880 / Nikon A1R HD25
- **Objective:** Plan Apo 63x / 1.40 NA oil immersion
- **Excitation:** 488 nm Argon laser, 5 mW
- **Pinhole:** 1.0 Airy Unit (AU) -- optimal for sectioning
- **Dwell time:** 2 us/pixel
- **Detection:** GaAsP PMT (Airyscan/spectral)
- **Frame interval:** 5 seconds

### Forward Model Pipeline

```
x (fluorescence distribution)
  |
  v
SourceNode: photon_source -- excitation scaling (1.0)
  |
  v
Element 1 (transport): conv2d -- confocal PSF convolution (sigma=1.0-1.2 px)
  |
  v
SensorNode: photon_sensor -- QE=0.25-0.9, gain=1.0
  |
  v
NoiseNode: poisson_gaussian_sensor -- Poisson (peak=5000) + Gaussian (sigma=0.02)
  |
  v
y (confocal image)
```

### Key Physics

- The confocal PSF is the product of excitation and detection PSFs: `h_confocal(r) = h_exc(r) * h_det(r)`, yielding ~1.4x better lateral resolution than widefield.
- Theoretical lateral resolution: `0.37 * lambda / NA = 0.37 * 525 / 1.4 = 139 nm`.
- Cumulative optical throughput is only 12.1% (vs ~50% for widefield), making the system photon-starved in live-cell conditions.
- The 1 AU pinhole setting balances sectioning capability against signal throughput; smaller pinholes drastically reduce signal.
- For live cells, the dominant mismatch sources are: (a) specimen motion during scanning, (b) focal drift from thermal effects, (c) photobleaching reducing effective fluorescence over time.

### Identified Forward Model Gaps

1. **PSF stationarity assumption:** The current model uses a shift-invariant PSF, but live-cell motion creates spatially varying blur that violates this assumption. The working process document acknowledges this ("motion blur makes PSF non-stationary") but the benchmark does not yet include a non-stationary PSF primitive.
2. **Temporal coupling:** The 200-frame time-lapse structure is not exploited by current solvers. Temporal regularization could substantially improve reconstruction quality.
3. **Widefield fallback warning:** The platform notes that the widefield fallback PSF (sigma=2.0) over-blurs by 30-60% compared to the true confocal PSF (sigma~1.2-1.5). This is a known source of catastrophic reconstruction failure.

---

## 4. State of the Art -- Literature Review (2024-2025)

### Self-inspired Noise2Noise (SN2N) -- Nature Methods, Sep 2024

Lequyer et al. introduced SN2N, a self-supervised denoising method requiring only a single noisy frame for training. Demonstrated on multiple confocal-based super-resolution systems (SIM, STED, Airyscan) with 1-2 orders of magnitude improved photon efficiency. Fully competitive with supervised methods (CARE, Noise2Void) without needing paired training data. Directly applicable to live-cell confocal where ground truth is unavailable.

**Relevance to benchmark:** SN2N could serve as a strong self-supervised baseline, especially for the hidden tier where no training pairs are available.

Reference: <https://www.nature.com/articles/s41592-024-02400-9>

### ZS-DeconvNet (Zero-shot Deconvolution Network) -- Nature Communications, Jun 2024

Li et al. presented ZS-DeconvNet for zero-shot resolution enhancement (>1.5x beyond diffraction limit) at 10x lower fluorescence than standard super-resolution. Applicable to both confocal and widefield modalities without ground truth. Uses physics-informed loss with known PSF structure.

**Relevance to benchmark:** The zero-shot paradigm aligns with the blind reconstruction challenge format. Could achieve both denoising and super-resolution simultaneously.

Reference: <https://www.nature.com/articles/s41467-024-48575-9>

### Volume Tells: Dual Cycle-Consistent Diffusion (VTCD) -- arXiv, Mar 2025

Proposed a diffusion-model approach for simultaneous 3D confocal denoising and super-resolution using intra-volume priors in a self-supervised framework. Demonstrates that diffusion models can leverage structural priors within a single volume without external training data.

**Relevance to benchmark:** While focused on 3D, the cycle-consistency framework could be adapted for temporal confocal sequences (200 frames). Represents the frontier of generative-model-based microscopy reconstruction.

Reference: <https://arxiv.org/html/2503.02261v1>

### Deep Learning in Fluorescence Imaging -- Wiley, 2024

Mao et al. provided a comprehensive review of deep learning for fluorescence microscopy including confocal, covering denoising (Noise2Void, DnCNN, CARE), super-resolution (RCAN, DFCAN), and segmentation. De-Abe enhances confocal image quality without additional optical components.

Reference: <https://onlinelibrary.wiley.com/doi/full/10.1002/jim4.17>

### Deep Learning Image Restoration for Fluorescence Microscopy -- Springer, 2025

Overview and resources for deep learning-based restoration and super-resolution in fluorescence microscopy, including practical protocols for confocal data.

Reference: <https://link.springer.com/protocol/10.1007/978-1-0716-4414-0_3>

### Gap Analysis vs. Current Leaderboard

| Capability | Current Benchmark | State of the Art |
|------------|-------------------|-----------------|
| Best method | Restormer (supervised, 2022) | SN2N (self-supervised, 2024) |
| Training requirement | Requires paired data | Single-frame or zero-shot |
| Super-resolution | Not addressed | 1.5x beyond diffraction (ZS-DeconvNet) |
| Temporal exploitation | None (frame-by-frame) | Temporal priors, cycle consistency |
| Diffusion models | Absent | VTCD (2025) for joint denoise+SR |
| Physics-informed DL | Gradient correction only | PSF-embedded loss functions |

The benchmark leaderboard is missing entries from the 2024-2025 generation of self-supervised and zero-shot methods. Adding SN2N and ZS-DeconvNet baselines would modernize the benchmark significantly.

---

## 5. Local Codebase & Data Integrity

### Files Inventory

| Path | Status | Notes |
|------|--------|-------|
| `benchmarks/configs/confocal_livecell.yaml` | Present | Core config: 512x512, C-->D DAG, 4 mismatch params |
| `benchmarks/expanded_configs/confocal_livecell_expanded.yaml` | Present | Expanded config with image sizes (128-1024), noise levels, 252 total cases |
| `docs/modality_benchmarks/confocal_livecell.md` | Present | B1-B4 benchmark specification at M0-M4 maturity levels |
| `docs/confocal_livecell_working_process.md` | Present | Comprehensive 820-line E2E pipeline trace (ideal + real scenarios) |
| `pwm/reports/confocal_livecell.md` | Present | PWM report with W1/W2 workflows, all 8 quick gates PASS |
| `scripts/run_confocal_livecell_experiment.py` | Present | Experiment runner script |
| `tests/test_casepack_confocal_livecell.py` | Present | Casepack unit test |
| `packages/pwm_core/contrib/casepacks/confocal_livecell_lowdose_drift_v1.json` | Present | Low-dose + drift casepack (400 photons, PnP-HQS solver) |
| `examples/specs/confocal_livecell_spec.json` | Present | Example spec (1000 photons, TV-FISTA solver) |
| `platform/.../confocal_livecell_b1_public.json` | Present | Public tier B1 data |
| `platform/.../confocal_livecell_b2_public.h5` | Present | Public tier B2 HDF5 |
| `platform/.../confocal_livecell_b3_public.tar.gz` | Present | Public tier B3 archive |
| `platform/.../confocal_livecell_b4_public.h5` | Present | Public tier B4 HDF5 |
| `datasets/benchmark/confocal_livecell/` | MISSING | No local dataset directory; benchmark relies on GCS-hosted data |

### Maturity Assessment

- **Current maturity:** M1 (single-parameter synthetic mismatch)
- **Benchmark tiers defined:** B1 (Design, 12 cases), B2 (Forward+Reconstruct, 80 cases), B3 (System ID, 80 cases), B4 (Correct+Diagnose, 80 cases) = 252 total
- **Solvers registered:** Richardson-Lucy (CPU), CARE (GPU, 2M params)
- **Metrics:** PSNR (primary), SSIM
- **Data source:** DeepBacs fluorescence (Zenodo, CC-BY-4.0) with synthetic phantom fallback

### Issues Found

1. **Missing local dataset:** The `datasets/benchmark/confocal_livecell/` directory does not exist. While platform data is hosted on GCS, local development and testing require a downloaded copy or a clear data-fetch script.
2. **Solver CARE duplication:** In the config YAML, `best_quality`, `famous_dl`, and `small_gpu` all point to the same `care_restore_2d` function but with inconsistent `params` ("2M", "2M", "0") and `gpu` flags (true, false, false). The `small_gpu` entry has `gpu: false` which is contradictory.
3. **Reference PSNR missing:** Both `reference_psnr` and `expected_psnr_range` are `null` in the config. The working process document establishes 26.1 dB (RL) and 30.2 dB (CARE) as reference values -- these should be propagated to the config.
4. **Mismatch parameter divergence:** Platform lists 3 parameters (pinhole diameter, refractive index, photobleaching rate); local config lists 4 different parameters (PSF sigma, drift rate, bleaching rate, pinhole misalignment); mismatch_db.yaml lists yet another set (psf_sigma, defocus, background, gain). These need harmonization.

---

## 6. Recommendations & Action Items

### Critical (must fix)

1. **Harmonize mismatch parameters** across platform page, `confocal_livecell.yaml`, `confocal_livecell_expanded.yaml`, and `mismatch_db.yaml`. Decide on a canonical set (the platform's 3-parameter model or the local 4-parameter model) and propagate everywhere.
2. **Populate `reference_psnr` and `expected_psnr_range`** in `confocal_livecell.yaml`. Suggested values: `reference_psnr: 26.1` (RL baseline), `expected_psnr_range: [22.0, 35.0]`.
3. **Fix solver config inconsistencies**: correct `small_gpu.gpu` to `true`, differentiate `famous_dl` from `best_quality` (e.g., Noise2Void vs. CARE), and fill in missing `reference` fields with paper citations.

### High Priority (should fix soon)

4. **Add SN2N baseline** (Lequyer et al., Nat. Methods 2024) to the leaderboard. This self-supervised method is directly applicable and would represent the current state of the art for training-data-free confocal denoising.
5. **Add ZS-DeconvNet baseline** (Li et al., Nat. Commun. 2024) as a zero-shot physics-informed entry.
6. **Create local dataset download script** or document the procedure to populate `datasets/benchmark/confocal_livecell/` from GCS for local development.
7. **Implement temporal regularization** for the 200-frame time-lapse data. Current solvers process frames independently, ignoring strong temporal correlations.

### Medium Priority (roadmap)

8. **Advance maturity to M2** by implementing compound mismatch (3+ simultaneous parameters) -- the expanded config already defines M2 but the benchmark report shows only M1 testing.
9. **Add non-stationary PSF primitive** to handle motion-induced spatially varying blur in live-cell data.
10. **Integrate Restormer as a local solver** in the solver registry -- it leads the leaderboard but is not registered in the local config.
11. **Add super-resolution evaluation metrics** (e.g., Fourier Ring Correlation) alongside PSNR/SSIM to measure resolution improvement beyond deconvolution.
12. **Expand real-data grounding** beyond DeepBacs: consider BioSR confocal subset and Cell Tracking Challenge confocal sequences for M3 maturity.

---

*Comprehensive 6-point review on 2026-03-03: platform page validated with all assets loading correctly (24/24 gallery, HDF5 on GCS confirmed); leaderboard analyzed showing Restormer leading at 0.730 overall with notable 9.65 dB Public-to-Hidden drop; forward model physics verified (confocal PSF product, 12.1% throughput, Poisson-Gaussian noise) with gaps identified in PSF stationarity and temporal coupling; literature survey covering SN2N (Nat. Methods 2024), ZS-DeconvNet (Nat. Commun. 2024), and VTCD (arXiv 2025) reveals benchmark is missing self-supervised and zero-shot baselines from the 2024-2025 generation; local codebase has 14+ confocal_livecell files across configs, docs, reports, scripts, casepacks, and platform data but missing local dataset directory and showing mismatch parameter divergence across 3 registries; 12 action items identified spanning critical config harmonization, high-priority baseline additions, and medium-priority maturity advancement toward M2/M3.*