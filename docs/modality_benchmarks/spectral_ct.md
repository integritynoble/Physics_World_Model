# Photon-Counting Spectral CT (`spectral_ct`)

**Category**: Medical Imaging | **Canonical DAG**: Pi --> W --> D | **Carrier**: X-ray
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: material_decomposition

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Energy bins, threshold settings, pile-up correction |
| **M1** Synthetic | Prompt tested with synthetic data validation: Energy bins, threshold settings, pile-up correction |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Energy bins, threshold settings, pile-up correction |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Energy bins, threshold settings, pile-up correction |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Photon-Counting Spectral CT |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Material decomposition under threshold drift, charge sharing |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Material decomposition under threshold drift, charge sharing |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Material decomposition under threshold drift, charge sharing |
| **M3** Real Data | Real experimental data with measured mismatch: Material decomposition under threshold drift, charge sharing |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Material decomposition under threshold drift, charge sharing |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Energy threshold calibration | 0 | [-2, 2] | keV per bin |
| Charge sharing fraction | 0 | [0, 15%] | - |
| Pile-up at high flux | 0 | [0, 10%] | - |
| Material decomposition basis error | 0 | [0, 5%] | - |

### Solvers & Expected Performance
- **Solver**: material_decomposition

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for Pi --> W --> D: Estimate energy thresholds, charge sharing width |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate energy thresholds, charge sharing width |
| **M2** Compound | Compound parameter identification (3+ params): Estimate energy thresholds, charge sharing width |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate energy thresholds, charge sharing width |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate energy thresholds, charge sharing width |

### True-Spec Parameters
Energy thresholds, charge sharing model, pile-up model, basis functions

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct threshold calibration, pile-up |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct threshold calibration, pile-up |
| **M2** Compound | Compound correction with rho measurement: Correct threshold calibration, pile-up |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct threshold calibration, pile-up |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct threshold calibration, pile-up |

### Correction Targets
- **Expected rho**: >= 0.75

### Improvement Roadmap
K-edge subtraction benchmark, multi-material decomposition.

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
