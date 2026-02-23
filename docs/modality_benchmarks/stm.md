# Scanning Tunneling Microscopy (STM) (`stm`)

**Category**: Scanning Probe Microscopy | **Canonical DAG**: S --> D | **Carrier**: Electron
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: ldos_normalization

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Bias voltage, tunneling current setpoint, scan speed |
| **M1** Synthetic | Prompt tested with synthetic data validation: Bias voltage, tunneling current setpoint, scan speed |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Bias voltage, tunneling current setpoint, scan speed |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Bias voltage, tunneling current setpoint, scan speed |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Scanning Tunneling Microscopy (STM) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Topography/LDOS under piezo hysteresis, thermal drift |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Topography/LDOS under piezo hysteresis, thermal drift |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Topography/LDOS under piezo hysteresis, thermal drift |
| **M3** Real Data | Real experimental data with measured mismatch: Topography/LDOS under piezo hysteresis, thermal drift |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Topography/LDOS under piezo hysteresis, thermal drift |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Tip electronic structure | ideal | variable LDOS | - |
| Piezo creep | 0 | [0, 5%] | - |
| Tunneling barrier height | 4.5 | [3.0, 6.0] | eV |
| Vibration amplitude | 0 | [0, 5] | pm |

### Solvers & Expected Performance
- **Solver**: ldos_normalization

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for S --> D: Estimate piezo coefficients, tip electronic state |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate piezo coefficients, tip electronic state |
| **M2** Compound | Compound parameter identification (3+ params): Estimate piezo coefficients, tip electronic state |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate piezo coefficients, tip electronic state |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate piezo coefficients, tip electronic state |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct piezo, flatten background, normalize LDOS |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct piezo, flatten background, normalize LDOS |
| **M2** Compound | Compound correction with rho measurement: Correct piezo, flatten background, normalize LDOS |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct piezo, flatten background, normalize LDOS |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct piezo, flatten background, normalize LDOS |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
