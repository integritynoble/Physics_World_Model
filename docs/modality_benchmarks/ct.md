# X-ray Computed Tomography (CT) (`ct`)

**Category**: Medical Imaging | **Canonical DAG**: Pi --> D | **Carrier**: X-ray
**Current Maturity**: M3 | **Forward Model**: linear_operator | **Default Solver**: fbp

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Geometry (fan/parallel/cone), angles, detector count, dose |
| **M1** Synthetic | Prompt tested with synthetic data validation: Geometry (fan/parallel/cone), angles, detector count, dose |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Geometry (fan/parallel/cone), angles, detector count, dose |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Geometry (fan/parallel/cone), angles, detector count, dose |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for X-ray Computed Tomography (CT) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: FBP/SART under center-of-rotation offset, angular offset, detector tilt, beam hardening |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: FBP/SART under center-of-rotation offset, angular offset, detector tilt, beam hardening |
| **M2** Compound | Compound mismatch (3+ params simultaneously): FBP/SART under center-of-rotation offset, angular offset, detector tilt, beam hardening |
| **M3** Real Data | Real experimental data with measured mismatch: FBP/SART under center-of-rotation offset, angular offset, detector tilt, beam hardening |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: FBP/SART under center-of-rotation offset, angular offset, detector tilt, beam hardening |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Center-of-rotation offset | 0 | [-5, 5] | px |
| Angular offset | 0 | [-3, 3] | deg |
| Detector tilt | 0 | [-2, 2] | deg |
| Beam hardening coeff | 0 | [0, 0.05] | - |
| Ring artifact amplitude | 0 | [0, 50] | counts |

### Solvers & Expected Performance
- **Solver**: fbp
- **Validated baseline**: FBP +10.68 dB, rho = 1.00

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for Pi --> D: Estimate CoR offset, angular errors, hardening coefficients |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate CoR offset, angular errors, hardening coefficients |
| **M2** Compound | Compound parameter identification (3+ params): Estimate CoR offset, angular errors, hardening coefficients |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate CoR offset, angular errors, hardening coefficients |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate CoR offset, angular errors, hardening coefficients |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct geometry; rho=100%, +10.68 dB |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct geometry; rho=100%, +10.68 dB |
| **M2** Compound | Compound correction with rho measurement: Correct geometry; rho=100%, +10.68 dB |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct geometry; rho=100%, +10.68 dB |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct geometry; rho=100%, +10.68 dB |

### Correction Targets
- **Expected rho**: 1.00

### Improvement Roadmap
Metal artifact reduction, limited-angle, scatter correction.

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
