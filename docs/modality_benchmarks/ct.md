# X-ray Computed Tomography (CT) (`ct`)

**Category**: Medical Imaging | **Canonical DAG**: Pi → D | **Carrier**: X-ray
**Current Maturity**: M3 | **Forward Model**: linear_operator | **Default Solver**: fbp

---

## Challenge Benchmark Results

### Dataset

| Tier | Scenes | Source | Content |
|------|--------|--------|---------|
| **Public** | 11 | [LoDoPaB-CT (Zenodo 3384092)](https://zenodo.org/records/3384092) | Real chest slices (LIDC/IDRI) |
| **Dev** | 20 | Procedural phantoms (GCS) | 5 anatomical backgrounds, round-robin |
| **Hidden** | 20 | Procedural phantoms + adversarial (GCS) | Metal inserts, lesions, calcifications |

Dev phantom types (`seed % 5`): `chest_upper`, `chest_mid`, `chest_lower`, `abdomen_upper`, `abdomen_mid`.

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
| `platform/.../benchmark_database/_algorithm_catalog.py` | 8-algorithm override + `CATEGORY_REAL_SCORES["ct"]` |
| `platform/.../benchmark_database/_challenge_data.py` | `CHALLENGE_CONFIG["ct"]`: tiers, mismatch ranges, baselines |
| `scripts/setup_benchmark_data.sh` | Downloads CT from GCS + LoDoPaB-CT from Zenodo (~1.5 GB) |
| `docs/Multi_Server_Setup.md` | Server D (CT) deployment guide |

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Geometry (fan/parallel/cone), angles, detector count, dose |
| **M1** Synthetic | Prompt tested with synthetic data validation: Geometry (fan/parallel/cone), angles, detector count, dose |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Geometry (fan/parallel/cone), angles, detector count, dose |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Geometry (fan/parallel/cone), angles, detector count, dose |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for X-ray Computed Tomography (CT) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: FBP/SART under center-of-rotation offset, angular offset, detector tilt, beam hardening |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: FBP/SART under center-of-rotation offset, angular offset, detector tilt, beam hardening |
| **M2** Compound | Compound mismatch (3+ params simultaneously): FBP/SART under center-of-rotation offset, angular offset, detector tilt, beam hardening |
| **M3** Real Data | Real experimental data with measured mismatch: FBP/SART under center-of-rotation offset, angular offset, detector tilt, beam hardening |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: FBP/SART under center-of-rotation offset, angular offset, detector tilt, beam hardening |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Center-of-rotation offset | 0 | [-5, 5] | px |
| Angular offset | 0 | [-3, 3] | deg |
| Detector tilt | 0 | [-2, 2] | deg |
| Beam hardening coeff | 0 | [0, 0.05] | - |
| Ring artifact amplitude | 0 | [0, 50] | counts |

### Solvers & Expected Performance
- **Solver**: fbp
- **Validated baseline**: FBP +10.68 dB, rho = 1.00

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for Pi --> D: Estimate CoR offset, angular errors, hardening coefficients |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate CoR offset, angular errors, hardening coefficients |
| **M2** Compound | Compound parameter identification (3+ params): Estimate CoR offset, angular errors, hardening coefficients |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate CoR offset, angular errors, hardening coefficients |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate CoR offset, angular errors, hardening coefficients |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct geometry; rho=100%, +10.68 dB |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct geometry; rho=100%, +10.68 dB |
| **M2** Compound | Compound correction with rho measurement: Correct geometry; rho=100%, +10.68 dB |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct geometry; rho=100%, +10.68 dB |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct geometry; rho=100%, +10.68 dB |

### Correction Targets
- **Expected rho**: 1.00

### Improvement Roadmap
Metal artifact reduction, limited-angle, scatter correction.

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
