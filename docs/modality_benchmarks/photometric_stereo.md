# Photometric Stereo (`photometric_stereo`)

**Category**: Depth Imaging | **Canonical DAG**: M --> C --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: normal_estimation

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Light source count, placement, surface BRDF |
| **M1** Synthetic | Prompt tested with synthetic data validation: Light source count, placement, surface BRDF |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Light source count, placement, surface BRDF |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Light source count, placement, surface BRDF |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Photometric Stereo |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Normal estimation under light position error |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Normal estimation under light position error |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Normal estimation under light position error |
| **M3** Real Data | Real experimental data with measured mismatch: Normal estimation under light position error |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Normal estimation under light position error |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Light direction error | 0 | [0, 5] | deg per source |
| Light intensity calibration | 1.0 | [0.8, 1.2] per source | - |
| Non-Lambertian surface fraction | 0 | [0, 30%] | - |
| Cast shadow fraction | 0 | [0, 15%] | of pixels |

### Solvers & Expected Performance
- **Solver**: normal_estimation

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> C --> D: Estimate light directions, surface albedo, BRDF |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate light directions, surface albedo, BRDF |
| **M2** Compound | Compound parameter identification (3+ params): Estimate light directions, surface albedo, BRDF |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate light directions, surface albedo, BRDF |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate light directions, surface albedo, BRDF |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct light calibration, inter-reflection |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct light calibration, inter-reflection |
| **M2** Compound | Compound correction with rho measurement: Correct light calibration, inter-reflection |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct light calibration, inter-reflection |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct light calibration, inter-reflection |

### Correction Targets
- **Expected rho**: TBD

### Improvement Roadmap
Add near-field photometric stereo, uncalibrated photometric stereo.

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
