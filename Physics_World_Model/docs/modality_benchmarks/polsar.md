# Polarimetric SAR (PolSAR) (`polsar`)

**Category**: Remote Sensing | **Canonical DAG**: F --> M --> D | **Carrier**: RF
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: polarimetric_decomposition

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Polarization modes, calibration targets |
| **M1** Synthetic | Prompt tested with synthetic data validation: Polarization modes, calibration targets |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Polarization modes, calibration targets |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Polarization modes, calibration targets |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Polarimetric SAR (PolSAR) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Polarimetric decomposition under cross-pol leakage |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Polarimetric decomposition under cross-pol leakage |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Polarimetric decomposition under cross-pol leakage |
| **M3** Real Data | Real experimental data with measured mismatch: Polarimetric decomposition under cross-pol leakage |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Polarimetric decomposition under cross-pol leakage |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Cross-talk between polarizations | 0 | [0, -25] | dB |
| Channel imbalance | 0 | [0, 1] | dB |
| Faraday rotation | 0 | [0, 5] | deg |

### Solvers & Expected Performance
- **Solver**: polarimetric_decomposition

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for F --> M --> D: Estimate cross-pol isolation, channel imbalance |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate cross-pol isolation, channel imbalance |
| **M2** Compound | Compound parameter identification (3+ params): Estimate cross-pol isolation, channel imbalance |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate cross-pol isolation, channel imbalance |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate cross-pol isolation, channel imbalance |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct polarimetric calibration, leakage |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct polarimetric calibration, leakage |
| **M2** Compound | Compound correction with rho measurement: Correct polarimetric calibration, leakage |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct polarimetric calibration, leakage |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct polarimetric calibration, leakage |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
