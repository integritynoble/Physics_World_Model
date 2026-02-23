# Ocean Acoustic Tomography (`ocean_acoustic_tomo`)

**Category**: Broader Experimental Science | **Canonical DAG**: P --> D | **Carrier**: Acoustic
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: travel_time_inversion

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Source-receiver transects, frequency, ray paths |
| **M1** Synthetic | Prompt tested with synthetic data validation: Source-receiver transects, frequency, ray paths |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Source-receiver transects, frequency, ray paths |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Source-receiver transects, frequency, ray paths |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Ocean Acoustic Tomography |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Travel-time inversion under sound speed profile error |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Travel-time inversion under sound speed profile error |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Travel-time inversion under sound speed profile error |
| **M3** Real Data | Real experimental data with measured mismatch: Travel-time inversion under sound speed profile error |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Travel-time inversion under sound speed profile error |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Sound speed profile error | 0 | [-2%, 2%] | - |
| Multipath identification | correct | [0, 20%] misassigned | - |
| Source/receiver position | 0 | [0, 10] | m |
| Current velocity error | 0 | [-0.5, 0.5] | m/s |

### Solvers & Expected Performance
- **Solver**: travel_time_inversion

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for P --> D: Estimate sound speed profile, current velocity |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate sound speed profile, current velocity |
| **M2** Compound | Compound parameter identification (3+ params): Estimate sound speed profile, current velocity |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate sound speed profile, current velocity |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate sound speed profile, current velocity |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct sound speed, ray bending model |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct sound speed, ray bending model |
| **M2** Compound | Compound correction with rho measurement: Correct sound speed, ray bending model |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct sound speed, ray bending model |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct sound speed, ray bending model |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
