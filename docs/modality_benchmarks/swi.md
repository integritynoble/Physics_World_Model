# Susceptibility-Weighted Imaging (SWI) (`swi`)

**Category**: Medical Imaging | **Canonical DAG**: M --> F --> S --> D | **Carrier**: Spin/RF
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: swi_phase_mask

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: TE, filter size, phase mask |
| **M1** Synthetic | Prompt tested with synthetic data validation: TE, filter size, phase mask |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for TE, filter size, phase mask |
| **M3** Real Data | Grounded in real experimental/clinical protocols: TE, filter size, phase mask |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Susceptibility-Weighted Imaging (SWI) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: SWI reconstruction under field inhomogeneity |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: SWI reconstruction under field inhomogeneity |
| **M2** Compound | Compound mismatch (3+ params simultaneously): SWI reconstruction under field inhomogeneity |
| **M3** Real Data | Real experimental data with measured mismatch: SWI reconstruction under field inhomogeneity |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: SWI reconstruction under field inhomogeneity |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Phase unwrapping error | 0 | [0, 5%] of voxels | - |
| Background field removal error | 0 | [0, 10%] | - |
| Dipole inversion regularization | optimal | +/- 50% | - |

### Solvers & Expected Performance
- **Solver**: swi_phase_mask

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> F --> S --> D: Estimate susceptibility sources, field map |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate susceptibility sources, field map |
| **M2** Compound | Compound parameter identification (3+ params): Estimate susceptibility sources, field map |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate susceptibility sources, field map |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate susceptibility sources, field map |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct background field, phase unwrapping |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct background field, phase unwrapping |
| **M2** Compound | Compound correction with rho measurement: Correct background field, phase unwrapping |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct background field, phase unwrapping |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct background field, phase unwrapping |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
