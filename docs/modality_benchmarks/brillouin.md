# Brillouin Microscopy (`brillouin`)

**Category**: Spectroscopy & Spectral Imaging | **Canonical DAG**: M --> R --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: lorentz_fit

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: VIPA etalon, extinction, NA, integration time |
| **M1** Synthetic | Prompt tested with synthetic data validation: VIPA etalon, extinction, NA, integration time |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for VIPA etalon, extinction, NA, integration time |
| **M3** Real Data | Grounded in real experimental/clinical protocols: VIPA etalon, extinction, NA, integration time |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Brillouin Microscopy |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Brillouin shift extraction under elastic background |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Brillouin shift extraction under elastic background |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Brillouin shift extraction under elastic background |
| **M3** Real Data | Real experimental data with measured mismatch: Brillouin shift extraction under elastic background |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Brillouin shift extraction under elastic background |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Brillouin shift calibration | 0 | [-50, 50] | MHz |
| VIPA FSR error | 0 | [-0.5%, 0.5%] | - |
| Elastic scattering leakage | 0 | [0, -30] dB | - |

### Solvers & Expected Performance
- **Solver**: lorentz_fit

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> R --> D: Estimate elastic leakage, free spectral range |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate elastic leakage, free spectral range |
| **M2** Compound | Compound parameter identification (3+ params): Estimate elastic leakage, free spectral range |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate elastic leakage, free spectral range |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate elastic leakage, free spectral range |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct elastic contamination, FSR calibration |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct elastic contamination, FSR calibration |
| **M2** Compound | Compound correction with rho measurement: Correct elastic contamination, FSR calibration |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct elastic contamination, FSR calibration |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct elastic contamination, FSR calibration |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
