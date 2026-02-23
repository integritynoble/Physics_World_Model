# Portal Imaging (EPID) (`portal_imaging`)

**Category**: Medical Imaging | **Canonical DAG**: Pi --> D | **Carrier**: MV X-ray
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: back_projection

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Detector geometry, gantry angle, field size |
| **M1** Synthetic | Prompt tested with synthetic data validation: Detector geometry, gantry angle, field size |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Detector geometry, gantry angle, field size |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Detector geometry, gantry angle, field size |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Portal Imaging (EPID) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Backprojection under sag, flex, MLC position error |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Backprojection under sag, flex, MLC position error |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Backprojection under sag, flex, MLC position error |
| **M3** Real Data | Real experimental data with measured mismatch: Backprojection under sag, flex, MLC position error |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Backprojection under sag, flex, MLC position error |

### Mismatch Parameters
Pi→D, MV X-ray. Gantry sag [0,3] mm, flex [0,5] mm, MLC error [-1,1] mm.

### Solvers & Expected Performance
- **Solver**: back_projection

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for Pi --> D: Estimate gantry sag, detector offset, MLC positions |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate gantry sag, detector offset, MLC positions |
| **M2** Compound | Compound parameter identification (3+ params): Estimate gantry sag, detector offset, MLC positions |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate gantry sag, detector offset, MLC positions |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate gantry sag, detector offset, MLC positions |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct geometric calibration, MLC model |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct geometric calibration, MLC model |
| **M2** Compound | Compound correction with rho measurement: Correct geometric calibration, MLC model |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct geometric calibration, MLC model |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct geometric calibration, MLC model |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
