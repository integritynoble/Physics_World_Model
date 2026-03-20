# TIRF Microscopy (`tirf`)

**Category**: Microscopy | **Canonical DAG**: C --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: richardson_lucy

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Incidence angle, evanescent depth, NA |
| **M1** Synthetic | Prompt tested with synthetic data validation: Incidence angle, evanescent depth, NA |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Incidence angle, evanescent depth, NA |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Incidence angle, evanescent depth, NA |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for TIRF Microscopy |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Deconvolution with evanescent-field PSF |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Deconvolution with evanescent-field PSF |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Deconvolution with evanescent-field PSF |
| **M3** Real Data | Real experimental data with measured mismatch: Deconvolution with evanescent-field PSF |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Deconvolution with evanescent-field PSF |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Incidence angle | 68 | [62, 75] | deg |
| Evanescent depth | 100 | [50, 300] | nm |
| Background (non-TIRF) | 0 | [0, 0.3] | relative |

### Solvers & Expected Performance
- **Solver**: richardson_lucy

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for C --> D: Estimate penetration depth, angle, background |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate penetration depth, angle, background |
| **M2** Compound | Compound parameter identification (3+ params): Estimate penetration depth, angle, background |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate penetration depth, angle, background |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate penetration depth, angle, background |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct angle calibration, background subtraction |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct angle calibration, background subtraction |
| **M2** Compound | Compound correction with rho measurement: Correct angle calibration, background subtraction |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct angle calibration, background subtraction |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct angle calibration, background subtraction |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
