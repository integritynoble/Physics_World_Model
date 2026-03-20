# Electrical Impedance Tomography (EIT) (`impedance_tomo`)

**Category**: Broader Experimental Science | **Canonical DAG**: M --> D | **Carrier**: Electric
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: gauss_newton

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Electrode count, current pattern, protocol |
| **M1** Synthetic | Prompt tested with synthetic data validation: Electrode count, current pattern, protocol |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Electrode count, current pattern, protocol |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Electrode count, current pattern, protocol |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Electrical Impedance Tomography (EIT) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Gauss-Newton under contact impedance error |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Gauss-Newton under contact impedance error |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Gauss-Newton under contact impedance error |
| **M3** Real Data | Real experimental data with measured mismatch: Gauss-Newton under contact impedance error |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Gauss-Newton under contact impedance error |

### Mismatch Parameters
M→D. Contact impedance [50,500] ohm, electrode position [0,5] mm.

### Solvers & Expected Performance
- **Solver**: gauss_newton

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> D: Estimate contact impedance, electrode positions |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate contact impedance, electrode positions |
| **M2** Compound | Compound parameter identification (3+ params): Estimate contact impedance, electrode positions |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate contact impedance, electrode positions |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate contact impedance, electrode positions |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct contact impedance, electrode model |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct contact impedance, electrode model |
| **M2** Compound | Compound correction with rho measurement: Correct contact impedance, electrode model |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct contact impedance, electrode model |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct contact impedance, electrode model |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
