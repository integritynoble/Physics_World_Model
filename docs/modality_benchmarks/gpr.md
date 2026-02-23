# Ground-Penetrating Radar (GPR) (`gpr`)

**Category**: Remote Sensing | **Canonical DAG**: P --> D | **Carrier**: RF
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: migration

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Antenna frequency, scan spacing, time window |
| **M1** Synthetic | Prompt tested with synthetic data validation: Antenna frequency, scan spacing, time window |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Antenna frequency, scan spacing, time window |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Antenna frequency, scan spacing, time window |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Ground-Penetrating Radar (GPR) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Migration under velocity model error, clutter |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Migration under velocity model error, clutter |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Migration under velocity model error, clutter |
| **M3** Real Data | Real experimental data with measured mismatch: Migration under velocity model error, clutter |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Migration under velocity model error, clutter |

### Mismatch Parameters
P→D, RF. Permittivity +/-20%, clutter [0,-10] dB.

### Solvers & Expected Performance
- **Solver**: migration

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for P --> D: Estimate permittivity profile, clutter model |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate permittivity profile, clutter model |
| **M2** Compound | Compound parameter identification (3+ params): Estimate permittivity profile, clutter model |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate permittivity profile, clutter model |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate permittivity profile, clutter model |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct velocity model, clutter suppression |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct velocity model, clutter suppression |
| **M2** Compound | Compound correction with rho measurement: Correct velocity model, clutter suppression |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct velocity model, clutter suppression |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct velocity model, clutter suppression |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
