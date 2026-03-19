# Sonar Imaging (`sonar`)

**Category**: Remote Sensing | **Canonical DAG**: P --> D | **Carrier**: Acoustic
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: beamform_das

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Transducer array, frequency, beamforming |
| **M1** Synthetic | Prompt tested with synthetic data validation: Transducer array, frequency, beamforming |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Transducer array, frequency, beamforming |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Transducer array, frequency, beamforming |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Sonar Imaging |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: DAS beamforming under sound speed error, multipath |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: DAS beamforming under sound speed error, multipath |
| **M2** Compound | Compound mismatch (3+ params simultaneously): DAS beamforming under sound speed error, multipath |
| **M3** Real Data | Real experimental data with measured mismatch: DAS beamforming under sound speed error, multipath |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: DAS beamforming under sound speed error, multipath |

### Mismatch Parameters
P→D, Acoustic. Speed +/-2%, multipath 1-3 paths.

### Solvers & Expected Performance
- **Solver**: beamform_das

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for P --> D: Estimate sound speed, multipath structure |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate sound speed, multipath structure |
| **M2** Compound | Compound parameter identification (3+ params): Estimate sound speed, multipath structure |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate sound speed, multipath structure |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate sound speed, multipath structure |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct sound speed, suppress multipath |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct sound speed, suppress multipath |
| **M2** Compound | Compound correction with rho measurement: Correct sound speed, suppress multipath |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct sound speed, suppress multipath |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct sound speed, suppress multipath |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
