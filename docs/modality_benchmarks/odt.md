# Optical Diffraction Tomography (ODT) (`odt`)

**Category**: Coherent Imaging | **Canonical DAG**: P --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: rytov_inversion

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Illumination angles, refractive index range, NA |
| **M1** Synthetic | Prompt tested with synthetic data validation: Illumination angles, refractive index range, NA |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Illumination angles, refractive index range, NA |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Illumination angles, refractive index range, NA |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Optical Diffraction Tomography (ODT) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Rytov/Born inversion under missing cone, RI drift |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Rytov/Born inversion under missing cone, RI drift |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Rytov/Born inversion under missing cone, RI drift |
| **M3** Real Data | Real experimental data with measured mismatch: Rytov/Born inversion under missing cone, RI drift |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Rytov/Born inversion under missing cone, RI drift |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Illumination angle error | 0 | [-2, 2] | deg per angle |
| Missing cone artifact | 30 | [20, 50] | deg |
| Refractive index of medium | 1.337 | [1.33, 1.35] | - |
| Multiple scattering | none | [0, 10%] | - |

### Solvers & Expected Performance
- **Solver**: rytov_inversion

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for P --> D: Estimate RI map, illumination angle errors |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate RI map, illumination angle errors |
| **M2** Compound | Compound parameter identification (3+ params): Estimate RI map, illumination angle errors |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate RI map, illumination angle errors |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate RI map, illumination angle errors |

### True-Spec Parameters
Illumination angles, medium RI, sample 3D RI distribution

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct missing cone, RI calibration |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct missing cone, RI calibration |
| **M2** Compound | Compound correction with rho measurement: Correct missing cone, RI calibration |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct missing cone, RI calibration |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct missing cone, RI calibration |

### Correction Targets
- **Expected rho**: >= 0.70

### Improvement Roadmap
Add Rytov vs Born approximation comparison.

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
