# Polarization Microscopy (`polarization`)

**Category**: Microscopy | **Canonical DAG**: M --> C --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: pnp_hqs

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Analyzer angles, retardance range, Stokes config |
| **M1** Synthetic | Prompt tested with synthetic data validation: Analyzer angles, retardance range, Stokes config |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Analyzer angles, retardance range, Stokes config |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Analyzer angles, retardance range, Stokes config |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Polarization Microscopy |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Mueller matrix reconstruction under calibration error |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Mueller matrix reconstruction under calibration error |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Mueller matrix reconstruction under calibration error |
| **M3** Real Data | Real experimental data with measured mismatch: Mueller matrix reconstruction under calibration error |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Mueller matrix reconstruction under calibration error |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Analyzer angle offset | 0 | [-5, 5] | deg |
| Retardance offset | 0 | [-10, 10] | nm |
| Extinction ratio | 1e-4 | [1e-5, 1e-3] | - |

### Solvers & Expected Performance
- **Solver**: pnp_hqs

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> C --> D: Estimate retardance offset, polarizer extinction |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate retardance offset, polarizer extinction |
| **M2** Compound | Compound parameter identification (3+ params): Estimate retardance offset, polarizer extinction |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate retardance offset, polarizer extinction |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate retardance offset, polarizer extinction |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct polarization calibration |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct polarization calibration |
| **M2** Compound | Compound correction with rho measurement: Correct polarization calibration |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct polarization calibration |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct polarization calibration |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
