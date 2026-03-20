# Machine Vision / AOI (`machine_vision`)

**Category**: Industrial Inspection | **Canonical DAG**: C --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: defect_detection

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Lens, illumination, resolution, FOV |
| **M1** Synthetic | Prompt tested with synthetic data validation: Lens, illumination, resolution, FOV |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Lens, illumination, resolution, FOV |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Lens, illumination, resolution, FOV |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Machine Vision / AOI |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Defect detection under illumination non-uniformity |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Defect detection under illumination non-uniformity |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Defect detection under illumination non-uniformity |
| **M3** Real Data | Real experimental data with measured mismatch: Defect detection under illumination non-uniformity |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Defect detection under illumination non-uniformity |

### Mismatch Parameters
C→D. Illumination [0,20%], MTF [0.2,0.8], distortion [0,3%].

### Solvers & Expected Performance
- **Solver**: defect_detection

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for C --> D: Estimate illumination profile, MTF, distortion |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate illumination profile, MTF, distortion |
| **M2** Compound | Compound parameter identification (3+ params): Estimate illumination profile, MTF, distortion |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate illumination profile, MTF, distortion |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate illumination profile, MTF, distortion |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct flat-field, lens distortion, focus |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct flat-field, lens distortion, focus |
| **M2** Compound | Compound correction with rho measurement: Correct flat-field, lens distortion, focus |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct flat-field, lens distortion, focus |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct flat-field, lens distortion, focus |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
