# Lucky Imaging (`lucky_imaging`)

**Category**: Astronomy & Space Imaging | **Canonical DAG**: M --> C --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: shift_and_add

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Frame rate, selection percentage, registration method |
| **M1** Synthetic | Prompt tested with synthetic data validation: Frame rate, selection percentage, registration method |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Frame rate, selection percentage, registration method |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Frame rate, selection percentage, registration method |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Lucky Imaging |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Shift-and-add under anisoplanatism, variable seeing |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Shift-and-add under anisoplanatism, variable seeing |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Shift-and-add under anisoplanatism, variable seeing |
| **M3** Real Data | Real experimental data with measured mismatch: Shift-and-add under anisoplanatism, variable seeing |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Shift-and-add under anisoplanatism, variable seeing |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Fried parameter (r0) | 15 | [5, 25] | cm |
| Frame selection threshold | 10% | [1%, 50%] | - |
| Isoplanatic angle | 5 | [2, 10] | arcsec |
| Registration error | 0 | [0, 0.5] | px |

### Solvers & Expected Performance
- **Solver**: shift_and_add

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> C --> D: Estimate Fried parameter, isoplanatic angle, tip-tilt |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate Fried parameter, isoplanatic angle, tip-tilt |
| **M2** Compound | Compound parameter identification (3+ params): Estimate Fried parameter, isoplanatic angle, tip-tilt |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate Fried parameter, isoplanatic angle, tip-tilt |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate Fried parameter, isoplanatic angle, tip-tilt |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct tip-tilt, deconvolve residual PSF |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct tip-tilt, deconvolve residual PSF |
| **M2** Compound | Compound correction with rho measurement: Correct tip-tilt, deconvolve residual PSF |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct tip-tilt, deconvolve residual PSF |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct tip-tilt, deconvolve residual PSF |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
