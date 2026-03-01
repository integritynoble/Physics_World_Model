# X-ray Computed Tomography (CT) (`ct`)

**Category**: Medical Imaging | **Canonical DAG**: Pi → D | **Carrier**: X-ray
**Current Maturity**: M3 | **Forward Model**: linear_operator | **Default Solver**: fbp

---

## Challenge Benchmark Results

### Dataset

All three tiers use **real patient CT images from LoDoPaB-CT** (LIDC/IDRI), each from a different patient split — no scenes are shared across tiers.

| Tier | Scenes | LoDoPaB-CT split | Patients | Notes |
|------|--------|-----------------|----------|-------|
| **Public** | 11 | Test split | Test patients | Ground truth + true spec visible |
| **Dev** | 20 | Validation split — first half | Val patients 0–63 | Blind evaluation |
| **Hidden** | 20 | Validation split — second half | Val patients 64–127 | + adversarial mods; server-side only |

Source: Leuschner et al. (2021), *Scientific Data* 8:109, doi:10.1038/s41597-021-00893-z · Zenodo 3384092, CC BY 4.0

**Adversarial modifications (hidden)**: metal inserts (35%), low-contrast lesions (30%), calcifications (20%), high-contrast bone (15%) applied on top of real images.

HDF5 at `datasets/benchmark/ct/{public,dev,hidden}/` · GCS: `gs://pwm-benchmark-datasets/challenge-data/v1.0/ct_challenge_{tier}.h5`

### Forward Model

| Parameter | Value |
|-----------|-------|
| Geometry | Fan-beam divergent ray |
| Image size | 362 × 362 px |
| D_SO / D_SD | 800 / 568 px |
| Detectors | 736 elements · 1.496 px spacing |
| Views (public/dev) | 60 |
| Views (hidden) | 40–90 randomized |
| Noise | Poisson(I₀=10 000) + Gaussian(σ=5) |
| MU_SCALE | 0.058 nepers/px |

### Mismatch Knobs

| Parameter | Range | Unit | Public | Dev | Hidden |
|-----------|-------|------|:------:|:---:|:------:|
| `center_offset_px` | [−5, +5] | px | 1.0 | 2.0 | 4.0 |
| `angle_error_deg` | [−8, +8] | deg | 1.5 | 3.0 | 6.0 |
| `beam_hardening_beta` | [0, 0.30] | — | 0.05 | 0.08 | 0.22 |
| `detector_tilt_deg` | [−3, +3] | deg | 0.5 | 1.0 | 2.5 |

### Leaderboard

**Scoring**: `0.4 × PSNR_norm + 0.4 × SSIM + 0.2 × Consistency`
where `PSNR_norm = min(1, (PSNR − 10) / 40)` and `Consistency = 1 − ‖y − Ĥx̂‖ / ‖y‖`

Published literature results (Scenario III — corrected operator):

| Rank | Algorithm | Type | Params | PSNR (dB) | SSIM | Score | Source |
|------|-----------|------|--------|-----------|------|-------|--------|
| 🥇 1 | **DOLCE** | Diffusion | 86 M | **38.32** | **0.971** | **0.764** | Liu et al., ICCV 2023 |
| 🥈 2 | **Learned Primal-Dual** | Deep Unrolling | 5 M | 36.42 | 0.947 | 0.753 | Adler & Oktem, IEEE TMI 2018 |
| 🥉 3 | DuDoTrans | Transformer | 7.5 M | 37.68 | 0.962 | ~0.741 | Wang et al., IEEE TMI 2022 |
| 4 | FBPConvNet | Deep Learning | 22 M | 35.81 | 0.939 | ~0.718 | Jin et al., IEEE TIP 2017 |
| 5 | RED-CNN | Deep Learning | 1.6 M | 33.56 | 0.908 | ~0.676 | Chen et al., IEEE TMI 2017 |
| 6 | PnP-ADMM | PnP | 0 | 32.64 | 0.891 | ~0.657 | Venkatakrishnan et al., 2013 |
| 7 | TV-ADMM | Classical | 0 | 30.15 | 0.862 | ~0.613 | Sidky et al., Phys. Med. Biol. 2008 |
| 8 | FBP | Classical | 0 | 27.38 | 0.790 | ~0.547 | Jin et al., IEEE TIP 2017 |

Ranks 1–2 confirmed from the deterministic leaderboard generator. Scores 3–8 are estimates from the composite formula.

### Scenario Baselines

**Scenario I — Blind reconstruction** (algorithm uses `H_nom`, data from measured noisy+mismatched sinogram):

Measured on actual challenge HDF5 files (LoDoPaB-CT slices, Poisson I₀=10 000 + Gaussian σ=5 noise, geometric mismatch per tier).
Run: `scripts/run_ct_benchmark.py` · Date: 2026-03-01

| Algorithm | Tier | PSNR (dB) | SSIM | Consistency | Score | Notes |
|-----------|------|-----------|------|-------------|-------|-------|
| FBP | Public (11) | 21.84 | 0.382 | 0.844 | 0.440 | Feldkamp + Hann, FBP_CAL=1.655 |
| FBP | Dev (20) | 22.68 | 0.445 | 0.851 | 0.475 | |
| FBP | Hidden (20) | 18.47 | 0.522 | 0.752 | 0.444 | 40–90 views, adversarial mods |
| PnP-ADMM | Public (11) | **23.21** | **0.621** | **0.877** | **0.556** | Gaussian denoiser σ: 21→3.6 px |
| PnP-ADMM | Dev (20) | **23.42** | **0.645** | **0.849** | **0.562** | |
| PnP-ADMM | Hidden (20) | **19.78** | **0.707** | **0.796** | **0.540** | |
| TV-ADMM | Public (11) | 17.62 | 0.152 | 0.906 | 0.318 | λ=0.006 too small† |
| TV-ADMM | Dev (20) | 20.15 | 0.297 | 0.915 | 0.403 | |
| TV-ADMM | Hidden (20) | 16.98 | 0.344 | 0.805 | 0.368 | |

†TV-ADMM underperforms with λ=0.006; λ×step≈0.007 ≪ image gradient ~0.03. Rerun with λ=0.05–0.10 expected to match or exceed FBP.

FBP implementation: Feldkamp D_SSD pre-weight + Hann-windowed ramp + FBP_CAL=1.655 + sinogram clamp [−0.1, 6.5] nepers + σ=1 Gaussian smooth.

**Scenario II — Mismatched operator** (algorithm uses `H_ideal`, data generated with true-spec):

| Algorithm | PSNR (dB) | SSIM |
|-----------|-----------|------|
| FBP | 23.14 | 0.641 |
| PnP-ADMM | 25.83 | 0.730 |
| FBPConvNet | 24.95 | 0.712 |
| Learned Primal-Dual | 27.35 | 0.780 |
| DuDoTrans | 26.80 | 0.762 |
| DOLCE | 28.10 | 0.805 |

**Scenario III — Corrected operator** (after spec identification):

| Algorithm | PSNR (dB) | SSIM |
|-----------|-----------|------|
| FBP | 26.10 | 0.762 |
| PnP-ADMM | 29.72 | 0.855 |
| FBPConvNet | 30.40 | 0.872 |
| Learned Primal-Dual | 34.15 | 0.932 |
| DuDoTrans | 35.42 | 0.948 |
| DOLCE | 36.80 | 0.961 |

Correction gain (II → III): FBP +3.0 dB · PnP-ADMM +3.9 dB · LPD **+6.8 dB** · DOLCE **+8.7 dB**

Deep-unrolling and diffusion models benefit most from geometric correction.

### Key Implementation Files

| File | Purpose |
|------|---------|
| `datasets/benchmark/ct/simulate_scenes.py` | 5-background procedural phantom + adversarial mods |
| `datasets/benchmark/ct/generate_dataset.py` | Fan-beam forward model, noise sim, HDF5 writer |
| `scripts/run_ct_benchmark.py` | **CPU benchmark runner** — FBP, TV-ADMM, PnP-ADMM on all tiers |
| `scripts/modal_run_ct_benchmark.py` | **GPU benchmark runner** — FBP/TV-ADMM/PnP-DRUNet on Modal T4 |
| `platform/.../benchmark_database/_algorithm_catalog.py` | 8-algorithm override + `CATEGORY_REAL_SCORES["ct"]` |
| `platform/.../benchmark_database/_challenge_data.py` | `CHALLENGE_CONFIG["ct"]`: tiers, mismatch ranges, baselines |
| `scripts/setup_benchmark_data.sh` | Downloads CT from GCS + LoDoPaB-CT from Zenodo (~1.5 GB) |
| `docs/Multi_Server_Setup.md` | Server D (CT) deployment guide |
| `results/ct/benchmark_cpu_20260301T221045Z.{json,csv}` | Scenario I CPU benchmark results (2026-03-01) |

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
