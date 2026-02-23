# X-ray Radiography (`xray_radiography`)

**Category**: Medical Imaging | **Canonical DAG**: Pi --> D | **Carrier**: X-ray
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: tv_fista

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Source-detector distance, filtration, exposure parameters |
| **M1** Synthetic | Prompt tested with synthetic data validation: Source-detector distance, filtration, exposure parameters |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Source-detector distance, filtration, exposure parameters |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Source-detector distance, filtration, exposure parameters |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for X-ray Radiography |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: TV-FISTA under scatter, beam hardening, detector lag |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: TV-FISTA under scatter, beam hardening, detector lag |
| **M2** Compound | Compound mismatch (3+ params simultaneously): TV-FISTA under scatter, beam hardening, detector lag |
| **M3** Real Data | Real experimental data with measured mismatch: TV-FISTA under scatter, beam hardening, detector lag |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: TV-FISTA under scatter, beam hardening, detector lag |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Scatter fraction | 0 | [0, 0.4] | - |
| Beam hardening | none | polynomial order 2-4 | - |
| Detector lag | 0 | [0, 0.1] | fraction |

### Solvers & Expected Performance
- **Solver**: tv_fista

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for Pi --> D: Estimate scatter fraction, hardening polynomial |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate scatter fraction, hardening polynomial |
| **M2** Compound | Compound parameter identification (3+ params): Estimate scatter fraction, hardening polynomial |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate scatter fraction, hardening polynomial |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate scatter fraction, hardening polynomial |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct scatter, beam hardening correction |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct scatter, beam hardening correction |
| **M2** Compound | Compound correction with rho measurement: Correct scatter, beam hardening correction |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct scatter, beam hardening correction |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct scatter, beam hardening correction |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
