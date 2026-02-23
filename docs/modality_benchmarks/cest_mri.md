# CEST MRI (`cest_mri`)

**Category**: Medical Imaging | **Canonical DAG**: M --> F --> S --> D | **Carrier**: Spin/RF
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: z_spectrum_fit

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Saturation frequency, power, B0 mapping |
| **M1** Synthetic | Prompt tested with synthetic data validation: Saturation frequency, power, B0 mapping |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Saturation frequency, power, B0 mapping |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Saturation frequency, power, B0 mapping |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for CEST MRI |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Z-spectrum fitting under B0/B1 inhomogeneity |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Z-spectrum fitting under B0/B1 inhomogeneity |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Z-spectrum fitting under B0/B1 inhomogeneity |
| **M3** Real Data | Real experimental data with measured mismatch: Z-spectrum fitting under B0/B1 inhomogeneity |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Z-spectrum fitting under B0/B1 inhomogeneity |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| B0 inhomogeneity | 0 | [-50, 50] | Hz |
| B1 inhomogeneity | 0 | [0, 20%] | - |
| Saturation power error | 0 | [-10%, 10%] | - |
| MT contamination | 0 | [0, 30%] | - |

### Solvers & Expected Performance
- **Solver**: z_spectrum_fit

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> F --> S --> D: Estimate B0 map, B1 map, CEST asymmetry |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate B0 map, B1 map, CEST asymmetry |
| **M2** Compound | Compound parameter identification (3+ params): Estimate B0 map, B1 map, CEST asymmetry |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate B0 map, B1 map, CEST asymmetry |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate B0 map, B1 map, CEST asymmetry |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct B0/B1 correction, water reference |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct B0/B1 correction, water reference |
| **M2** Compound | Compound correction with rho measurement: Correct B0/B1 correction, water reference |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct B0/B1 correction, water reference |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct B0/B1 correction, water reference |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
