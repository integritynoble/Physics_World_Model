# MINFLUX Nanoscopy (`minflux`)

**Category**: Microscopy | **Canonical DAG**: C --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: mle_localization

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Beam pattern, localization precision, photon budget |
| **M1** Synthetic | Prompt tested with synthetic data validation: Beam pattern, localization precision, photon budget |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Beam pattern, localization precision, photon budget |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Beam pattern, localization precision, photon budget |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for MINFLUX Nanoscopy |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Localization under beam misalignment |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Localization under beam misalignment |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Localization under beam misalignment |
| **M3** Real Data | Real experimental data with measured mismatch: Localization under beam misalignment |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Localization under beam misalignment |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Beam center error | 0 | [0, 5] | nm |
| Photon count | 500 | [50, 2000] | photons |

### Solvers & Expected Performance
- **Solver**: mle_localization

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for C --> D: Estimate beam center position, photon rate |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate beam center position, photon rate |
| **M2** Compound | Compound parameter identification (3+ params): Estimate beam center position, photon rate |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate beam center position, photon rate |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate beam center position, photon rate |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct beam positioning errors |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct beam positioning errors |
| **M2** Compound | Compound correction with rho measurement: Correct beam positioning errors |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct beam positioning errors |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct beam positioning errors |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
