# Cathodoluminescence (CL) Imaging (`cathodoluminescence`)

**Category**: Scientific Instrumentation | **Canonical DAG**: M --> R --> D | **Carrier**: Electron
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: hyperspectral_unmixing

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Beam energy, spectral range, collection optics |
| **M1** Synthetic | Prompt tested with synthetic data validation: Beam energy, spectral range, collection optics |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Beam energy, spectral range, collection optics |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Beam energy, spectral range, collection optics |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Cathodoluminescence (CL) Imaging |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Hyperspectral mapping under drift, beam damage |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Hyperspectral mapping under drift, beam damage |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Hyperspectral mapping under drift, beam damage |
| **M3** Real Data | Real experimental data with measured mismatch: Hyperspectral mapping under drift, beam damage |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Hyperspectral mapping under drift, beam damage |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Beam current drift | 0 | [0, 5%] | - |
| Collection efficiency variation | 0 | [0, 20%] | spatial |
| Spectral calibration error | 0 | [-2, 2] | nm |
| Carbon contamination | 0 | [0, 10%] | signal loss |

### Solvers & Expected Performance
- **Solver**: hyperspectral_unmixing

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> R --> D: Estimate spectral response, damage rate, drift |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate spectral response, damage rate, drift |
| **M2** Compound | Compound parameter identification (3+ params): Estimate spectral response, damage rate, drift |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate spectral response, damage rate, drift |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate spectral response, damage rate, drift |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct drift, damage model, spectral calibration |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct drift, damage model, spectral calibration |
| **M2** Compound | Compound correction with rho measurement: Correct drift, damage model, spectral calibration |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct drift, damage model, spectral calibration |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct drift, damage model, spectral calibration |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
