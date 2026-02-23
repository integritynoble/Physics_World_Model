# Image Scanning Microscopy (ISM) (`ism`)

**Category**: Microscopy | **Canonical DAG**: C --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: pixel_reassignment

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Detector array geometry, reassignment strategy |
| **M1** Synthetic | Prompt tested with synthetic data validation: Detector array geometry, reassignment strategy |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Detector array geometry, reassignment strategy |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Detector array geometry, reassignment strategy |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Image Scanning Microscopy (ISM) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Pixel reassignment under geometric distortion |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Pixel reassignment under geometric distortion |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Pixel reassignment under geometric distortion |
| **M3** Real Data | Real experimental data with measured mismatch: Pixel reassignment under geometric distortion |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Pixel reassignment under geometric distortion |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Detector element offset | 0 | [-1, 1] | px |
| Magnification error | 0 | [-5%, 5%] | relative |

### Solvers & Expected Performance
- **Solver**: pixel_reassignment

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for C --> D: Estimate detector offset, magnification error |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate detector offset, magnification error |
| **M2** Compound | Compound parameter identification (3+ params): Estimate detector offset, magnification error |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate detector offset, magnification error |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate detector offset, magnification error |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct pixel reassignment parameters |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct pixel reassignment parameters |
| **M2** Compound | Compound correction with rho measurement: Correct pixel reassignment parameters |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct pixel reassignment parameters |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct pixel reassignment parameters |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
