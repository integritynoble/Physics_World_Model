# STED Microscopy (`sted`)

**Category**: Microscopy | **Canonical DAG**: C --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: richardson_lucy

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Depletion beam shape, saturation power, resolution |
| **M1** Synthetic | Prompt tested with synthetic data validation: Depletion beam shape, saturation power, resolution |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Depletion beam shape, saturation power, resolution |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Depletion beam shape, saturation power, resolution |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for STED Microscopy |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Deconvolution with effective PSF under STED mismatch |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Deconvolution with effective PSF under STED mismatch |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Deconvolution with effective PSF under STED mismatch |
| **M3** Real Data | Real experimental data with measured mismatch: Deconvolution with effective PSF under STED mismatch |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Deconvolution with effective PSF under STED mismatch |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Depletion beam alignment | 0 | [0, 30] | nm offset |
| Saturation factor | 30 | [10, 50] | - |
| Effective PSF FWHM | 40 | [30, 120] | nm |

### Solvers & Expected Performance
- **Solver**: richardson_lucy

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for C --> D: Estimate depletion efficiency, effective resolution |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate depletion efficiency, effective resolution |
| **M2** Compound | Compound parameter identification (3+ params): Estimate depletion efficiency, effective resolution |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate depletion efficiency, effective resolution |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate depletion efficiency, effective resolution |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct depletion beam alignment |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct depletion beam alignment |
| **M2** Compound | Compound correction with rho measurement: Correct depletion beam alignment |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct depletion beam alignment |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct depletion beam alignment |

### Correction Targets
- **Expected rho**: >= 0.70

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
