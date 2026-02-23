# Magnetic Resonance Imaging (MRI) (`mri`)

**Category**: Medical Imaging | **Canonical DAG**: M --> F --> S --> D | **Carrier**: Spin/RF
**Current Maturity**: M3 | **Forward Model**: linear_operator | **Default Solver**: sense

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Coil count, trajectory (Cartesian/radial/spiral), acceleration factor |
| **M1** Synthetic | Prompt tested with synthetic data validation: Coil count, trajectory (Cartesian/radial/spiral), acceleration factor |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Coil count, trajectory (Cartesian/radial/spiral), acceleration factor |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Coil count, trajectory (Cartesian/radial/spiral), acceleration factor |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Magnetic Resonance Imaging (MRI) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: SENSE/GRAPPA under coil sensitivity error, k-space trajectory deviation, off-resonance |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: SENSE/GRAPPA under coil sensitivity error, k-space trajectory deviation, off-resonance |
| **M2** Compound | Compound mismatch (3+ params simultaneously): SENSE/GRAPPA under coil sensitivity error, k-space trajectory deviation, off-resonance |
| **M3** Real Data | Real experimental data with measured mismatch: SENSE/GRAPPA under coil sensitivity error, k-space trajectory deviation, off-resonance |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: SENSE/GRAPPA under coil sensitivity error, k-space trajectory deviation, off-resonance |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Coil sensitivity error | 0 | [0, 15%] per coil | relative |
| k-space trajectory deviation | 0 | [0, 2%] | - |
| Off-resonance (B0) | 0 | [-100, 100] | Hz |
| Acceleration factor | R=4 | [2, 8] | - |

### Solvers & Expected Performance
- **Solver**: sense
- **Validated baseline**: SENSE +1.75 to +7.14 dB

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> F --> S --> D: Estimate coil maps, trajectory errors, field map |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate coil maps, trajectory errors, field map |
| **M2** Compound | Compound parameter identification (3+ params): Estimate coil maps, trajectory errors, field map |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate coil maps, trajectory errors, field map |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate coil maps, trajectory errors, field map |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct coil + trajectory; +1.75-7.14 dB |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct coil + trajectory; +1.75-7.14 dB |
| **M2** Compound | Compound correction with rho measurement: Correct coil + trajectory; +1.75-7.14 dB |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct coil + trajectory; +1.75-7.14 dB |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct coil + trajectory; +1.75-7.14 dB |

### Correction Targets
- **Expected rho**: TBD

### Improvement Roadmap
Non-Cartesian, R=8/R=16, phase error estimation.

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
