# PALM/STORM Single-Molecule Localization (`palm_storm`)

**Category**: Microscopy | **Canonical DAG**: M --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: thunderstorm

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Label density, photon budget, frame count |
| **M1** Synthetic | Prompt tested with synthetic data validation: Label density, photon budget, frame count |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Label density, photon budget, frame count |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Label density, photon budget, frame count |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for PALM/STORM Single-Molecule Localization |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Localization under drift and background mismatch |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Localization under drift and background mismatch |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Localization under drift and background mismatch |
| **M3** Real Data | Real experimental data with measured mismatch: Localization under drift and background mismatch |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Localization under drift and background mismatch |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Drift rate (x, y) | 0 | [0, 2] each | nm/frame |
| Background photons | 20 | [5, 100] | per px |
| Photon count/event | 1000 | [200, 5000] | photons |
| Pixel size | 100 | [90, 110] | nm |

### Solvers & Expected Performance
- **Solver**: thunderstorm

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> D: Estimate drift trajectory, photon rate, background |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate drift trajectory, photon rate, background |
| **M2** Compound | Compound parameter identification (3+ params): Estimate drift trajectory, photon rate, background |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate drift trajectory, photon rate, background |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate drift trajectory, photon rate, background |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct drift, re-localize with updated model |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct drift, re-localize with updated model |
| **M2** Compound | Compound correction with rho measurement: Correct drift, re-localize with updated model |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct drift, re-localize with updated model |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct drift, re-localize with updated model |

### Correction Targets
- **Expected rho**: >= 0.80

### Improvement Roadmap
Add multi-emitter, 3D SMLM.

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
