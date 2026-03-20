# MR Angiography (MRA) (`mra`)

**Category**: Medical Imaging | **Canonical DAG**: M --> F --> S --> D | **Carrier**: Spin/RF
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: mip_recon

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: TOF vs CE vs PC, spatial resolution, coverage |
| **M1** Synthetic | Prompt tested with synthetic data validation: TOF vs CE vs PC, spatial resolution, coverage |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for TOF vs CE vs PC, spatial resolution, coverage |
| **M3** Real Data | Grounded in real experimental/clinical protocols: TOF vs CE vs PC, spatial resolution, coverage |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for MR Angiography (MRA) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: MIP/SSD under flow artifact, vessel signal |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: MIP/SSD under flow artifact, vessel signal |
| **M2** Compound | Compound mismatch (3+ params simultaneously): MIP/SSD under flow artifact, vessel signal |
| **M3** Real Data | Real experimental data with measured mismatch: MIP/SSD under flow artifact, vessel signal |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: MIP/SSD under flow artifact, vessel signal |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Contrast timing error | 0 | [-3, 3] | s |
| Background suppression | complete | [0, 20%] residual | - |
| Velocity encoding error | 0 | [-15%, 15%] | - |

### Solvers & Expected Performance
- **Solver**: mip_recon

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> F --> S --> D: Estimate flow velocity, vessel boundary |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate flow velocity, vessel boundary |
| **M2** Compound | Compound parameter identification (3+ params): Estimate flow velocity, vessel boundary |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate flow velocity, vessel boundary |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate flow velocity, vessel boundary |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct flow artifacts, background suppression |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct flow artifacts, background suppression |
| **M2** Compound | Compound correction with rho measurement: Correct flow artifacts, background suppression |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct flow artifacts, background suppression |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct flow artifacts, background suppression |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
