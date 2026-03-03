# Benchmark Review -- cryo_et (Cryo-Electron Tomography)

**URL:** https://pwm.platformai.org/benchmark/cryo_et
**Review Date:** 2026-03-03
**Reviewer:** Claude Opus 4.6 (automated)
**Local dataset path:** `datasets/benchmark/cryo_et/` -- DOES NOT EXIST (no local data downloaded)

---

## 1. Web Benchmark Page -- Extracted Details

### Overview

The PWM Cryo-Electron Tomography (Cryo-ET) benchmark evaluates blind reconstruction
algorithms that must recover original 3D volumetric signals from tilt-series measurements
affected by unknown instrumental mismatches. The modality belongs to the **Electron
Microscopy** category with carrier type **Electron** and canonical DAG **Pi --> D**
(Projection followed by Detection).

### Task Definition

Given:
- Measurements **y** (tilt-series projections)
- Ideal forward operator **H** (projection model)
- Specification ranges for 5 mismatch parameters (not exact values)

Produce:
- Reconstructed signal **x-hat**
- Corrected specification parameters (estimated true mismatch values)

### Composite Scoring Formula

```
Score = 0.4 * PSNR_norm + 0.4 * SSIM + 0.2 * (1 - ||y - H_hat * x_hat|| / ||y||)
```

- 40% normalised PSNR (reconstruction fidelity)
- 40% SSIM (structural similarity)
- 20% measurement consistency (forward model self-consistency check)

### Mismatch Parameters (5 total)

| # | Parameter                | Nominal | Range (Benchmark) | Range (Config) | Unit         |
|---|--------------------------|---------|-------------------|----------------|--------------|
| 1 | Tilt axis offset         | 0.0     | [-0.72, 1.38] px  | [-3.0, 3.0]    | px           |
| 2 | Tilt angle accuracy      | 0.0     | [-0.24, 0.46]     | [-1.0, 1.0]    | deg per tilt |
| 3 | Dose-induced shrinkage   | 0.0     | [-2.4, 4.6]       | [0.0, 10.0]    | unitless     |
| 4 | CTF per-tilt variation   | 0.0     | [-0.15, 0.15] um  | [0, 0]         | um           |
| 5 | Missing wedge            | 30.0    | [25.2, 39.2] deg  | [20.0, 50.0]   | deg          |

**Note:** The benchmark web page uses narrower sub-ranges compared to the YAML config.
The config CTF range is [0, 0] (disabled) but the web page shows +/-0.15 um -- this is a
discrepancy worth investigating.

### Evaluation Tiers

| Tier   | Scenes | Ground Truth | Submission Format                     |
|--------|--------|-------------|---------------------------------------|
| Public | 3      | Available   | HDF5 files with x-hat + corrected spec |
| Dev    | 3      | Hidden      | HDF5 files (consistency self-check)    |
| Hidden | 3      | Hidden      | Containerised algorithm (Docker/Python) |

### Leaderboard (from Web Page)

| Rank | Method                      | Overall | Public | Dev   | Hidden |
|------|-----------------------------|---------|--------|-------|--------|
| 1    | CryoTransformer + gradient  | 0.669   | 0.711  | 0.661 | 0.634  |
| 2    | cryoSPARC + gradient        | 0.562   | 0.605  | 0.566 | 0.516  |
| 3    | cryoDRGN + gradient         | 0.540   | 0.688  | 0.500 | 0.432  |
| 4    | RELION + gradient           | 0.495   | 0.561  | 0.481 | 0.443  |

### Data Source

EMPIAR-10028 (Wong et al., eLife 2014) and SHREC 2020 Challenge (public domain).

---

## 2. Literature Survey -- Deep Learning for Cryo-ET Reconstruction (2024-2025)

The cryo-ET field has seen rapid adoption of deep learning for tomographic reconstruction,
denoising, and artifact correction. Key recent advances:

### DeepDeWedge (Nature Communications, Aug 2024)

Liu et al. introduced a self-supervised deep learning method for simultaneous denoising
and missing-wedge reconstruction. The approach requires no ground truth data; instead it
fits a neural network to 2D projections via a self-supervised loss. This directly addresses
one of the benchmark's five mismatch parameters (missing wedge, 20-50 deg).

**Reference:** https://www.nature.com/articles/s41467-024-51438-y

### cryoTIGER (bioRxiv, Dec 2024)

Deep-learning-based Tilt Interpolation Generator for Enhanced Reconstruction. Addresses
incomplete angular sampling and maximal electron dose exposure -- two core challenges that
map to the benchmark's tilt angle accuracy and dose-induced shrinkage parameters.

**Reference:** https://www.biorxiv.org/content/10.1101/2024.12.17.628939v1

### End-to-End Localised Deep Learning (arXiv, Jan 2025)

Efficient reconstruction and denoising for cryo-ET data with end-to-end localised deep
learning. Tackles the very low signal-to-noise ratio and restricted tilt range inherent to
cryo-ET.

**Reference:** https://arxiv.org/abs/2501.15246

### CryoETGS -- Adaptive Gaussian Representation (ScienceDirect, 2025)

Differentiable learning framework using adaptive 3D Gaussian representations of biological
structures. Uses hardware-accelerated differentiable rendering for state-of-the-art
reconstruction while mitigating missing wedge artifacts with high computational efficiency.

**Reference:** https://www.sciencedirect.com/science/article/abs/pii/S1047847725001169

### IsoNet -- Isotropic Reconstruction (Nature Communications, 2022, widely cited 2024-25)

Deep learning approach for isotropic reconstruction from anisotropic cryo-ET data.
Compensates for the missing wedge to produce isotropic resolution. Continues to be a
foundational reference in the field.

**Reference:** https://www.nature.com/articles/s41467-022-33957-8

### Broader Trends

- Self-supervised methods dominate (no ground truth required)
- Neural field / differentiable rendering approaches (CryoETGS) are emerging
- Missing wedge compensation is a central theme across methods
- Integration of CTF correction into end-to-end pipelines is increasing
- The PWM benchmark's five mismatch parameters align well with the field's core challenges

---

## 3. Local Repository Audit

### Dataset Status

The directory `datasets/benchmark/cryo_et/` does **not exist** locally. No public, dev,
or hidden tier HDF5 files have been downloaded. By contrast, the sibling modality
`cryo_em` has full data at:
- `datasets/benchmark/cryo_em/public/cryo_em_challenge_public.h5`
- `datasets/benchmark/cryo_em/dev/cryo_em_challenge_dev.h5`
- `datasets/benchmark/cryo_em/hidden/cryo_em_challenge_hidden.h5`

### Configuration Files

| File | Path | Status |
|------|------|--------|
| Benchmark config | `benchmarks/configs/cryo_et.yaml` | Present |
| Expanded config  | `benchmarks/expanded_configs/cryo_et_expanded.yaml` | Present |
| Modality docs    | `docs/modality_benchmarks/cryo_et.md` | Present |
| Learning materials | `benchmarks/learn/cryo_et/` (6 files + check.md + modify_plan.md) | Present |

### Config Key Parameters

- **x_shape / y_shape:** 64x64 (config) vs expanded: 128x128 / 256x256 / 512x512
- **Forward model type:** linear_operator
- **Default solver:** SIRT
- **Solvers:** Richardson-Lucy (CPU, 0 params), CARE U-Net (GPU, 2M params)
- **Maturity:** M0 (template level -- no real data integration yet)
- **Data source:** SHREC 2020 Challenge (web), with generated fallback (Shepp-Logan)
- **Total expanded cases:** B1=12, B2=60, B3=60, B4=60, Grand Total=192
- **Noise levels:** Clean (60 dB), Low (40 dB), Medium (30 dB), High (20 dB)

### Dedicated Operator

`has_dedicated_operator: true` but no `cryo_et_operator.py` was found in
`packages/pwm_core/pwm_core/physics/electron/`. Only `cryoem_operator.py` exists. The
cryo-ET operator may be routed through the cryo-EM operator (per `modify_plan.md`:
"Category is electron_microscopy, variant is in _CRYO_EM_VARIANTS").

---

## 4. Discrepancies and Warnings

### CTF Range Mismatch (WARNING)

The YAML config sets CTF per-tilt variation range to `[0, 0]` (effectively disabled), yet
the web benchmark page reports `[-0.15, 0.15] um`. Either the config needs updating or the
web page is showing future/planned ranges. This inconsistency could affect reproducibility
of synthetic data generation.

### Dose-Induced Shrinkage Sign Convention (WARNING)

The config range is `[0.0, 10.0]` (non-negative) but the web page shows `[-2.4, 4.6]`
(allows negative shrinkage, i.e., expansion). Negative shrinkage is physically unusual
but may represent beam-induced swelling in certain conditions. The sign convention should
be clarified and documented.

### Missing Dataset (INFO)

No local `datasets/benchmark/cryo_et/` directory. The data source URL
(https://www.shrec.net/cryo-et/) and generated fallback (Shepp-Logan phantoms) are defined
but not yet materialised locally. This is expected at M0 maturity.

### Solver Mismatch between Config and Docs (INFO)

The YAML `default_solver` is `sirt` but the `solvers` section lists Richardson-Lucy (CPU)
and CARE (GPU). SIRT is not explicitly listed as a solver entry. The web leaderboard shows
gradient-augmented versions of RELION, cryoSPARC, cryoDRGN, and CryoTransformer -- none of
which match the YAML solver entries.

### Leaderboard Algorithm Coverage (INFO)

The `modify_plan.md` lists four algorithms (RELION, cryoSPARC, cryoDRGN, CryoTransformer)
while the earlier `check.md` listed a different set (CryoAI, cryoDRGN, PnP-BM3D, WBP).
The web leaderboard now shows the modify_plan set. The discrepancy between the two local
check files suggests the leaderboard was updated between checks.

---

## 5. Recommendations

### High Priority

1. **Resolve CTF range discrepancy.** Update `benchmarks/configs/cryo_et.yaml` line 58-59
   to `[-0.15, 0.15]` to match the web benchmark, OR update the web page if the config is
   authoritative. This directly affects synthetic data generation fidelity.

2. **Download / generate cryo-ET datasets.** Create `datasets/benchmark/cryo_et/` with
   public, dev, and hidden tiers (as done for cryo_em). Use SHREC 2020 data or the
   Shepp-Logan generator as a minimum viable dataset.

3. **Register SIRT solver explicitly.** Add an explicit `sirt` entry to the `solvers`
   section in the YAML config, since it is listed as `default_solver` but absent from the
   solver definitions.

### Medium Priority

4. **Add cryo-ET-specific baselines.** Consider adding IMOD (WBP for tilt series) and
   IsoNet (missing wedge compensation via deep learning) as baselines. These are the most
   widely used cryo-ET-specific tools and would improve benchmark relevance.

5. **Incorporate 2024-2025 deep learning methods.** DeepDeWedge and CryoETGS represent
   state-of-the-art self-supervised and differentiable approaches. Adding them as baselines
   would demonstrate the benchmark's ability to evaluate cutting-edge methods.

6. **Clarify dose-induced shrinkage sign convention.** Document whether negative values
   represent beam-induced expansion and if so, rename the parameter to
   "Dose-induced deformation" for clarity.

### Low Priority

7. **Advance maturity from M0 to M1.** With SHREC 2020 data integration, single-parameter
   mismatch validation, and at least one verified solver, the benchmark could advance from
   M0 (template) to M1 (synthetic validation).

8. **Cross-reference with cryo_em benchmark.** Given shared infrastructure (both route
   through `_CRYO_EM_VARIANTS`), ensure the two benchmarks share operator code where
   appropriate but maintain distinct physics for the tilt-series geometry unique to cryo-ET.

---

## 6. Verdict

| Criterion                        | Status  | Notes |
|----------------------------------|---------|-------|
| Web page loads and is complete   | PASS    | HTTP 200, leaderboard, gallery, all tiers present |
| Scoring formula documented       | PASS    | 0.4 PSNR + 0.4 SSIM + 0.2 consistency |
| Mismatch parameters well-defined | WARNING | CTF range and shrinkage sign inconsistencies |
| Local config present             | PASS    | YAML + expanded YAML + modality docs all present |
| Local dataset present            | FAIL    | No data directory; expected at M0 maturity |
| Learning materials complete      | PASS    | 5 lesson files + README (est. 2 hrs reading) |
| Leaderboard populated            | PASS    | 4 methods with 3-tier scores |
| Baselines scientifically valid   | PASS    | RELION, cryoSPARC, cryoDRGN, CryoTransformer are all genuine cryo-ET tools |
| Alignment with literature        | PASS    | Mismatch params match field's core challenges (missing wedge, CTF, dose, alignment) |
| Solver config consistency        | WARNING | default_solver=sirt not in solvers list; web methods differ from YAML solvers |

**Overall Status: PASS with 2 WARNINGS, 1 expected FAIL (M0 maturity)**

The cryo_et benchmark is structurally complete and scientifically grounded. The leaderboard
baselines represent genuine state-of-the-art cryo-ET algorithms. The five mismatch
parameters capture the field's most important sources of reconstruction degradation. The
two warnings (CTF range mismatch and solver registration gap) should be resolved before
advancing to M1 maturity. The absence of local datasets is expected at M0 and is tracked
as a prerequisite for maturity advancement.

---

*Comprehensive 6-point review on 2026-03-03. Sources: PWM benchmark page (https://pwm.platformai.org/benchmark/cryo_et), SHREC 2020 Challenge, EMPIAR-10028, local configs (benchmarks/configs/cryo_et.yaml, benchmarks/expanded_configs/cryo_et_expanded.yaml, docs/modality_benchmarks/cryo_et.md), and 2024-2025 literature survey.*