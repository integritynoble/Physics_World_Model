# STEM-EDX Elemental Mapping (`edx_mapping`)

**Category**: Electron Microscopy | **Canonical DAG**: M --> R --> D | **Carrier**: Electron
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: cliff_lorimer

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Beam current, dwell time, detector solid angle |
| **M1** Synthetic | Prompt tested with synthetic data validation: Beam current, dwell time, detector solid angle |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Beam current, dwell time, detector solid angle |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Beam current, dwell time, detector solid angle |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for STEM-EDX Elemental Mapping |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Element quantification under absorption, fluorescence |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Element quantification under absorption, fluorescence |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Element quantification under absorption, fluorescence |
| **M3** Real Data | Real experimental data with measured mismatch: Element quantification under absorption, fluorescence |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Element quantification under absorption, fluorescence |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Absorption correction error | 0 | [0, 15%] | - |
| Detector solid angle | measured | +/- 10% | sr |
| Peak overlap (spectral) | 0 | [0, 3] elements | - |
| Bremsstrahlung background | measured | +/- 20% | - |

### Solvers & Expected Performance
- **Solver**: cliff_lorimer

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> R --> D: Estimate absorption correction, fluorescence yield |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate absorption correction, fluorescence yield |
| **M2** Compound | Compound parameter identification (3+ params): Estimate absorption correction, fluorescence yield |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate absorption correction, fluorescence yield |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate absorption correction, fluorescence yield |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct absorption, cliff-lorimer factors |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct absorption, cliff-lorimer factors |
| **M2** Compound | Compound correction with rho measurement: Correct absorption, cliff-lorimer factors |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct absorption, cliff-lorimer factors |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct absorption, cliff-lorimer factors |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
