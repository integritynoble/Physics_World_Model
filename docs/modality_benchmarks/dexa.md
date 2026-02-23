# Dual-Energy X-ray Absorptiometry (DEXA) (`dexa`)

**Category**: Medical Imaging | **Canonical DAG**: Pi --> D | **Carrier**: X-ray
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: dual_energy_decomposition

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Dual energy selection, scan mode, calibration phantom |
| **M1** Synthetic | Prompt tested with synthetic data validation: Dual energy selection, scan mode, calibration phantom |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Dual energy selection, scan mode, calibration phantom |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Dual energy selection, scan mode, calibration phantom |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Dual-Energy X-ray Absorptiometry (DEXA) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Dual-energy decomposition under beam hardening, fat-lean mismatch |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Dual-energy decomposition under beam hardening, fat-lean mismatch |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Dual-energy decomposition under beam hardening, fat-lean mismatch |
| **M3** Real Data | Real experimental data with measured mismatch: Dual-energy decomposition under beam hardening, fat-lean mismatch |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Dual-energy decomposition under beam hardening, fat-lean mismatch |

### Mismatch Parameters
Pi → D, X-ray. Effective energies +/-15%, calibration +/-5%, fat fraction [0,20%].

### Solvers & Expected Performance
- **Solver**: dual_energy_decomposition

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for Pi --> D: Estimate effective energies, calibration polynomial |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate effective energies, calibration polynomial |
| **M2** Compound | Compound parameter identification (3+ params): Estimate effective energies, calibration polynomial |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate effective energies, calibration polynomial |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate effective energies, calibration polynomial |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct calibration, decomposition coefficients |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct calibration, decomposition coefficients |
| **M2** Compound | Compound correction with rho measurement: Correct calibration, decomposition coefficients |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct calibration, decomposition coefficients |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct calibration, decomposition coefficients |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
