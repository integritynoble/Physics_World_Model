# Ultrasonic Phased Array (TFM/FMC) (`ultrasonic_phased_array`)

**Category**: Industrial Inspection | **Canonical DAG**: P --> D | **Carrier**: Acoustic
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: total_focusing_method

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Element count, frequency, focal law, wedge angle |
| **M1** Synthetic | Prompt tested with synthetic data validation: Element count, frequency, focal law, wedge angle |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Element count, frequency, focal law, wedge angle |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Element count, frequency, focal law, wedge angle |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Ultrasonic Phased Array (TFM/FMC) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: TFM/FMC under velocity error, coupling variation |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: TFM/FMC under velocity error, coupling variation |
| **M2** Compound | Compound mismatch (3+ params simultaneously): TFM/FMC under velocity error, coupling variation |
| **M3** Real Data | Real experimental data with measured mismatch: TFM/FMC under velocity error, coupling variation |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: TFM/FMC under velocity error, coupling variation |

### Mismatch Parameters
P→D. Velocity +/-3%, coupling [0.5,1.5], wedge +/-2 deg.

### Solvers & Expected Performance
- **Solver**: total_focusing_method

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for P --> D: Estimate velocity, coupling, element sensitivity |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate velocity, coupling, element sensitivity |
| **M2** Compound | Compound parameter identification (3+ params): Estimate velocity, coupling, element sensitivity |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate velocity, coupling, element sensitivity |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate velocity, coupling, element sensitivity |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct velocity, element calibration |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct velocity, element calibration |
| **M2** Compound | Compound correction with rho measurement: Correct velocity, element calibration |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct velocity, element calibration |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct velocity, element calibration |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
