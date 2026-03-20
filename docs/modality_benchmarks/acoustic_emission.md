# Acoustic Emission Testing (AE) (`acoustic_emission`)

**Category**: Broader Experimental Science | **Canonical DAG**: P --> S --> D | **Carrier**: Acoustic
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: source_localization

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Sensor placement, frequency range, threshold |
| **M1** Synthetic | Prompt tested with synthetic data validation: Sensor placement, frequency range, threshold |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Sensor placement, frequency range, threshold |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Sensor placement, frequency range, threshold |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Acoustic Emission Testing (AE) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Source localization under velocity anisotropy |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Source localization under velocity anisotropy |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Source localization under velocity anisotropy |
| **M3** Real Data | Real experimental data with measured mismatch: Source localization under velocity anisotropy |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Source localization under velocity anisotropy |

### Mismatch Parameters
P→S→D. Velocity anisotropy [0,15%], coupling [0.5,1.5].

### Solvers & Expected Performance
- **Solver**: source_localization

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for P --> S --> D: Estimate wave velocity, coupling, source type |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate wave velocity, coupling, source type |
| **M2** Compound | Compound parameter identification (3+ params): Estimate wave velocity, coupling, source type |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate wave velocity, coupling, source type |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate wave velocity, coupling, source type |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct velocity model, localization |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct velocity model, localization |
| **M2** Compound | Compound correction with rho measurement: Correct velocity model, localization |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct velocity model, localization |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct velocity model, localization |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
