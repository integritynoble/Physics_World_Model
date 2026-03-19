# Intravascular Ultrasound (IVUS) (`ivus`)

**Category**: Medical Imaging | **Canonical DAG**: P --> D | **Carrier**: Acoustic
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: polar_recon

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Transducer frequency (20-60 MHz), pullback speed |
| **M1** Synthetic | Prompt tested with synthetic data validation: Transducer frequency (20-60 MHz), pullback speed |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Transducer frequency (20-60 MHz), pullback speed |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Transducer frequency (20-60 MHz), pullback speed |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Intravascular Ultrasound (IVUS) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Polar reconstruction under NURD, catheter eccentricity |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Polar reconstruction under NURD, catheter eccentricity |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Polar reconstruction under NURD, catheter eccentricity |
| **M3** Real Data | Real experimental data with measured mismatch: Polar reconstruction under NURD, catheter eccentricity |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Polar reconstruction under NURD, catheter eccentricity |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Catheter rotation non-uniformity | 0 | [0, 10%] | - |
| Ring-down artifact | 0 | [0, 20%] depth | - |
| Sound speed in plaque | 1540 | [1400, 1700] | m/s |

### Solvers & Expected Performance
- **Solver**: polar_recon

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for P --> D: Estimate NURD profile, catheter position offset |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate NURD profile, catheter position offset |
| **M2** Compound | Compound parameter identification (3+ params): Estimate NURD profile, catheter position offset |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate NURD profile, catheter position offset |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate NURD profile, catheter position offset |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct NURD, re-center catheter model |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct NURD, re-center catheter model |
| **M2** Compound | Compound correction with rho measurement: Correct NURD, re-center catheter model |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct NURD, re-center catheter model |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct NURD, re-center catheter model |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
