# Correlative Light-Electron Microscopy (CLEM) (`clem`)

**Category**: Multi-Modal Fusion | **Canonical DAG**: C --> D (LM) + C --> D (EM) --> Fusion | **Carrier**: Photon + Electron
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: overlay_registration

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Sample prep, fiducials, overlay registration |
| **M1** Synthetic | Prompt tested with synthetic data validation: Sample prep, fiducials, overlay registration |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Sample prep, fiducials, overlay registration |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Sample prep, fiducials, overlay registration |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Correlative Light-Electron Microscopy (CLEM) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Overlay under shrinkage, distortion, scale difference |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Overlay under shrinkage, distortion, scale difference |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Overlay under shrinkage, distortion, scale difference |
| **M3** Real Data | Real experimental data with measured mismatch: Overlay under shrinkage, distortion, scale difference |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Overlay under shrinkage, distortion, scale difference |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Registration error (LM to EM) | 0 | [0, 500] | nm |
| Sample deformation (fixation) | 0 | [0, 5%] | shrinkage |
| Fluorescence preservation | 100% | [30%, 100%] | - |

### Solvers & Expected Performance
- **Solver**: overlay_registration

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for C --> D (LM) + C --> D (EM) --> Fusion: Estimate scale factor, rotation, nonlinear distortion |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate scale factor, rotation, nonlinear distortion |
| **M2** Compound | Compound parameter identification (3+ params): Estimate scale factor, rotation, nonlinear distortion |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate scale factor, rotation, nonlinear distortion |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate scale factor, rotation, nonlinear distortion |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct registration, compensate shrinkage |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct registration, compensate shrinkage |
| **M2** Compound | Compound correction with rho measurement: Correct registration, compensate shrinkage |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct registration, compensate shrinkage |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct registration, compensate shrinkage |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
