# Pump-Probe Microscopy (`pump_probe`)

**Category**: Ultrafast Imaging | **Canonical DAG**: M --> R --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: transient_absorption

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Pump/probe wavelengths, delay range, repetition rate |
| **M1** Synthetic | Prompt tested with synthetic data validation: Pump/probe wavelengths, delay range, repetition rate |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Pump/probe wavelengths, delay range, repetition rate |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Pump/probe wavelengths, delay range, repetition rate |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Pump-Probe Microscopy |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Transient absorption recon under chirp, scatter |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Transient absorption recon under chirp, scatter |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Transient absorption recon under chirp, scatter |
| **M3** Real Data | Real experimental data with measured mismatch: Transient absorption recon under chirp, scatter |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Transient absorption recon under chirp, scatter |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Time-zero drift | 0 | [-100, 100] | fs |
| Pump power fluctuation | 0 | [0, 5%] | - |
| Chirp (GDD) | 0 | [-500, 500] | fs^2 |
| Spatial overlap error | 0 | [0, 20%] of beam | - |

### Solvers & Expected Performance
- **Solver**: transient_absorption

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> R --> D: Estimate chirp function, coherent artifact, scatter |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate chirp function, coherent artifact, scatter |
| **M2** Compound | Compound parameter identification (3+ params): Estimate chirp function, coherent artifact, scatter |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate chirp function, coherent artifact, scatter |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate chirp function, coherent artifact, scatter |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct chirp, coherent artifact subtraction |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct chirp, coherent artifact subtraction |
| **M2** Compound | Compound correction with rho measurement: Correct chirp, coherent artifact subtraction |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct chirp, coherent artifact subtraction |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct chirp, coherent artifact subtraction |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
