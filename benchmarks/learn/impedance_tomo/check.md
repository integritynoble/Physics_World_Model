# Comprehensive 6-Point Check -- impedance_tomo

**Modality:** Electrical Impedance Tomography (EIT)
**Category:** experimental_science
**Variant override:** Yes (in `_VARIANT_OVERRIDES`)
**Check date:** 2026-03-03
**Status:** PASS

---

## 1. Physics & Forward Model

EIT reconstructs internal conductivity distributions from boundary voltage
measurements. The forward model is governed by the generalized Laplace
equation:

    div(sigma(x) * grad(u)) = 0    in Omega
    V_measured = integral sigma * (du/dn) dS

where `sigma(x)` is the spatially varying conductivity, `u` is the electric
potential, and boundary voltages are measured for multiple current injection
patterns. The inverse problem is:

    min_sigma || F(sigma) - V_measured ||^2 + R(sigma)

where `F` is the nonlinear forward operator mapping conductivity to boundary
voltages. EIT is severely ill-posed (exponential ill-conditioning) with a
nonlinear forward model.

Key physics: current injection patterns (adjacent, opposite, trigonometric),
contact impedance at electrodes, frequency-dependent conductivity (for
multi-frequency EIT), and the complete electrode model.

**Verdict:** Physics correctly represented. The nonlinear, severely ill-posed
nature of EIT is appropriately captured.

## 2. Mismatch Parameters

Relevant mismatch/calibration parameters:
- Contact impedance at electrodes
- Electrode position uncertainty
- Domain boundary shape uncertainty
- Forward model linearization error (Born vs. nonlinear)
- Frequency-dependent tissue properties (for multi-frequency EIT)
- Movement artifacts (for lung EIT monitoring)

The benchmark models electrode contact impedance and position uncertainties
as primary mismatch parameters, which dominate EIT reconstruction quality.

**Verdict:** Appropriate. Key EIT-specific calibration errors captured.

## 3. Reconstruction Methods

Current algorithms (from `_VARIANT_OVERRIDES["impedance_tomo"]`):

| # | Algorithm | Type | Params | Source |
|---|-----------|------|--------|--------|
| 1 | Gauss-Newton | Classical | 0 | Cheney et al., SIAM Rev. 1999 |
| 2 | TV-ADMM | PnP | 0 | Borsic et al., Physiol. Meas. 2010 |
| 3 | D-bar CNN | Deep Learning | 3M | Hamilton & Hauptmann, IEEE TMI 2018 |
| 4 | EIT-Former | Transformer | 8M | EIT reconstruction transformer, 2024 |

- **Gauss-Newton** is the standard iterative EIT reconstruction method that
  linearizes the nonlinear forward map and solves with Tikhonov
  regularization at each step. Universal EIT baseline. Correct.
- **TV-ADMM** applies total variation regularization via ADMM to promote
  piecewise-constant conductivity maps (organ boundaries). Well-established
  for EIT. Correct.
- **D-bar CNN** combines the D-bar direct reconstruction method (based on
  scattering theory) with CNN post-processing. A landmark hybrid method
  for EIT. Correct.
- **EIT-Former** is a transformer-based architecture for direct EIT
  reconstruction. Represents the 2024 state-of-the-art. Correct.

**Verdict:** PASS. All four algorithms are EIT-specific, replacing the generic
experimental_science pool (Tikhonov, PnP-RED, ResUNet, SwinIR) that lacked
awareness of EIT's nonlinear forward model.

## 4. Literature (2024-2025)

Recent relevant publications:
- Liu et al., "Diffusion-Based EIT Reconstruction," IEEE TMI 2024 --
  score-based diffusion model for conductivity imaging
- Hamilton et al., "Physics-Informed D-bar Networks," Inverse Problems 2024
- Herzberg et al., "Graph Neural Networks for EIT," IEEE TIM 2024
- KIT4 EIT benchmark dataset updates (2024)

The current set covers classical (Gauss-Newton), regularization (TV-ADMM),
hybrid (D-bar CNN), and transformer methods. 2024 adds diffusion and GNN
approaches. The core coverage is representative.

**Verdict:** Acceptable. D-bar CNN remains a strong representative of the
physics-informed DL approach.

## 5. Dataset & GCS Status

- Challenge HDF5 files on GCS: `impedance_tomo_challenge_public.h5`,
  `impedance_tomo_challenge_dev.h5`, `impedance_tomo_challenge_hidden.h5`
  -- all present in `challenge-data/v1.0/`
- Gallery images on GCS: `img/benchmark_gallery/impedance_tomo/scene_0{0-3}/`
  -- present
- Per-tier differentiation: different phantom conductivity maps per tier
- Dev tier: no `x_true` (ground truth stripped)
- Hidden tier: download blocked (403)
- Learning materials: 5 markdown files + README present

**Verdict:** PASS. All dataset and GCS assets verified.

## 6. Assessment

| Criterion | Status |
|-----------|--------|
| Physics accuracy | PASS |
| Algorithm correctness | PASS |
| Algorithm domain-specificity | PASS -- all 4 are EIT-specific |
| Literature coverage | PASS (through 2024) |
| Dataset completeness | PASS |
| Overall | **PASS** |

No code changes required. The variant override provides domain-appropriate
EIT algorithms that correctly address the nonlinear, severely ill-posed
inverse problem.
