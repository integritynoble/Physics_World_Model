# Compressed Ultrafast Photography (CUP) (`cup`)

**Category**: Ultrafast Imaging | **Canonical DAG**: M --> Sigma --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: cup_recon

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | "Design T-CUP for light-in-flight: 10 trillion fps, 256x256 spatial." |
| **M1** Synthetic | Prompt tested with synthetic data validation: Streak speed, DMD pattern, spatial encoding |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Streak speed, DMD pattern, spatial encoding |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Streak speed, DMD pattern, spatial encoding |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Compressed Ultrafast Photography (CUP) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: CUP reconstruction under streak nonlinearity, DMD error |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: CUP reconstruction under streak nonlinearity, DMD error |
| **M2** Compound | Compound mismatch (3+ params simultaneously): CUP reconstruction under streak nonlinearity, DMD error |
| **M3** Real Data | Real experimental data with measured mismatch: CUP reconstruction under streak nonlinearity, DMD error |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: CUP reconstruction under streak nonlinearity, DMD error |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| DMD encoding error | 0 | [0, 2%] bit flip | - |
| Streak sweep calibration | 0 | [-5%, 5%] | - |
| Temporal-spatial coupling | 0 | [0, 10%] | - |

### Solvers & Expected Performance
- **Solver**: cup_recon

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> Sigma --> D: Estimate streak function, DMD misalignment |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate streak function, DMD misalignment |
| **M2** Compound | Compound parameter identification (3+ params): Estimate streak function, DMD misalignment |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate streak function, DMD misalignment |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate streak function, DMD misalignment |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct streak, DMD calibration |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct streak, DMD calibration |
| **M2** Compound | Compound correction with rho measurement: Correct streak, DMD calibration |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct streak, DMD calibration |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct streak, DMD calibration |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
