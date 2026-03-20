# Shear-Wave Elastography (`elastography`)

**Category**: Medical Imaging | **Canonical DAG**: P --> D | **Carrier**: Acoustic
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: time_of_flight_inversion

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Push pulse, tracking method, shear wave frequency |
| **M1** Synthetic | Prompt tested with synthetic data validation: Push pulse, tracking method, shear wave frequency |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Push pulse, tracking method, shear wave frequency |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Push pulse, tracking method, shear wave frequency |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Shear-Wave Elastography |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: TOF inversion under wave reflection, dispersion |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: TOF inversion under wave reflection, dispersion |
| **M2** Compound | Compound mismatch (3+ params simultaneously): TOF inversion under wave reflection, dispersion |
| **M3** Real Data | Real experimental data with measured mismatch: TOF inversion under wave reflection, dispersion |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: TOF inversion under wave reflection, dispersion |

### Mismatch Parameters
P→D, Acoustic. Speed error +/-20%, reflection [0,30%], dispersion [0,20%].

### Solvers & Expected Performance
- **Solver**: time_of_flight_inversion

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for P --> D: Estimate shear wave speed, attenuation, boundary effects |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate shear wave speed, attenuation, boundary effects |
| **M2** Compound | Compound parameter identification (3+ params): Estimate shear wave speed, attenuation, boundary effects |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate shear wave speed, attenuation, boundary effects |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate shear wave speed, attenuation, boundary effects |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct reflection, dispersion compensation |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct reflection, dispersion compensation |
| **M2** Compound | Compound correction with rho measurement: Correct reflection, dispersion compensation |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct reflection, dispersion compensation |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct reflection, dispersion compensation |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
