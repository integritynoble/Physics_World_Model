# Fiber Bundle Endoscopy (`endoscopy`)

**Category**: Medical Imaging | **Canonical DAG**: M --> C --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: tv_fista

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Fiber count, FOV, bending radius, illumination |
| **M1** Synthetic | Prompt tested with synthetic data validation: Fiber count, FOV, bending radius, illumination |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Fiber count, FOV, bending radius, illumination |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Fiber count, FOV, bending radius, illumination |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Fiber Bundle Endoscopy |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: TV-FISTA under fiber cross-talk, non-uniform transmission |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: TV-FISTA under fiber cross-talk, non-uniform transmission |
| **M2** Compound | Compound mismatch (3+ params simultaneously): TV-FISTA under fiber cross-talk, non-uniform transmission |
| **M3** Real Data | Real experimental data with measured mismatch: TV-FISTA under fiber cross-talk, non-uniform transmission |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: TV-FISTA under fiber cross-talk, non-uniform transmission |

### Mismatch Parameters
M→C→D, Photon. Fiber transmission [0,15%], distortion [0,5%], cross-talk [0,10%].

### Solvers & Expected Performance
- **Solver**: tv_fista

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> C --> D: Estimate fiber transmission map, geometric distortion |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate fiber transmission map, geometric distortion |
| **M2** Compound | Compound parameter identification (3+ params): Estimate fiber transmission map, geometric distortion |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate fiber transmission map, geometric distortion |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate fiber transmission map, geometric distortion |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct fiber calibration, distortion |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct fiber calibration, distortion |
| **M2** Compound | Compound correction with rho measurement: Correct fiber calibration, distortion |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct fiber calibration, distortion |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct fiber calibration, distortion |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
