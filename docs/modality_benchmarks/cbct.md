# Cone-Beam Computed Tomography (CBCT) (`cbct`)

**Category**: Medical Imaging | **Canonical DAG**: Pi --> D | **Carrier**: X-ray
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: fdk

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Cone angle, flat-panel geometry, rotation arc, dose |
| **M1** Synthetic | Prompt tested with synthetic data validation: Cone angle, flat-panel geometry, rotation arc, dose |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Cone angle, flat-panel geometry, rotation arc, dose |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Cone angle, flat-panel geometry, rotation arc, dose |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Cone-Beam Computed Tomography (CBCT) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: FDK under cone-beam artifacts, scatter, truncation |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: FDK under cone-beam artifacts, scatter, truncation |
| **M2** Compound | Compound mismatch (3+ params simultaneously): FDK under cone-beam artifacts, scatter, truncation |
| **M3** Real Data | Real experimental data with measured mismatch: FDK under cone-beam artifacts, scatter, truncation |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: FDK under cone-beam artifacts, scatter, truncation |

### Mismatch Parameters
Pi → D, X-ray. Scatter [0.2,0.7], truncation [0,20%], gantry flex [0,2] mm. rho >= 0.80.

### Solvers & Expected Performance
- **Solver**: fdk

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for Pi --> D: Estimate scatter fraction, truncation extent, detector offset |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate scatter fraction, truncation extent, detector offset |
| **M2** Compound | Compound parameter identification (3+ params): Estimate scatter fraction, truncation extent, detector offset |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate scatter fraction, truncation extent, detector offset |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate scatter fraction, truncation extent, detector offset |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct scatter, extend FOV, ring artifacts |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct scatter, extend FOV, ring artifacts |
| **M2** Compound | Compound correction with rho measurement: Correct scatter, extend FOV, ring artifacts |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct scatter, extend FOV, ring artifacts |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct scatter, extend FOV, ring artifacts |

### Correction Targets
- **Expected rho**: >= 0.80.

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
