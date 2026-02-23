# Magnetic Force Microscopy (MFM) (`mfm`)

**Category**: Scanning Probe Microscopy | **Canonical DAG**: S --> M --> D | **Carrier**: Magnetic
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: lift_mode_recon

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Lift height, tip magnetization, scan rate |
| **M1** Synthetic | Prompt tested with synthetic data validation: Lift height, tip magnetization, scan rate |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Lift height, tip magnetization, scan rate |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Lift height, tip magnetization, scan rate |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Magnetic Force Microscopy (MFM) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Magnetic recon under topographic crosstalk, tip degradation |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Magnetic recon under topographic crosstalk, tip degradation |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Magnetic recon under topographic crosstalk, tip degradation |
| **M3** Real Data | Real experimental data with measured mismatch: Magnetic recon under topographic crosstalk, tip degradation |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Magnetic recon under topographic crosstalk, tip degradation |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Lift height | 50 | [20, 200] | nm |
| Tip magnetization model | point dipole | +/- 30% moment | - |
| Electrostatic coupling | 0 | [0, 10%] | - |

### Solvers & Expected Performance
- **Solver**: lift_mode_recon

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for S --> M --> D: Estimate lift height variation, tip moment, crosstalk |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate lift height variation, tip moment, crosstalk |
| **M2** Compound | Compound parameter identification (3+ params): Estimate lift height variation, tip moment, crosstalk |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate lift height variation, tip moment, crosstalk |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate lift height variation, tip moment, crosstalk |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct lift height, topographic subtraction |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct lift height, topographic subtraction |
| **M2** Compound | Compound correction with rho measurement: Correct lift height, topographic subtraction |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct lift height, topographic subtraction |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct lift height, topographic subtraction |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
