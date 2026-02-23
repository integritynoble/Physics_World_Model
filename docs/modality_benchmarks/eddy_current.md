# Eddy Current Imaging (`eddy_current`)

**Category**: Industrial Inspection | **Canonical DAG**: F --> D | **Carrier**: EM
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: impedance_inversion

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Frequency, probe geometry, lift-off compensation |
| **M1** Synthetic | Prompt tested with synthetic data validation: Frequency, probe geometry, lift-off compensation |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Frequency, probe geometry, lift-off compensation |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Frequency, probe geometry, lift-off compensation |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Eddy Current Imaging |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Impedance map under lift-off variation |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Impedance map under lift-off variation |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Impedance map under lift-off variation |
| **M3** Real Data | Real experimental data with measured mismatch: Impedance map under lift-off variation |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Impedance map under lift-off variation |

### Mismatch Parameters
F→D. Lift-off [0,0.5] mm, conductivity +/-10%.

### Solvers & Expected Performance
- **Solver**: impedance_inversion

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for F --> D: Estimate lift-off, conductivity, probe alignment |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate lift-off, conductivity, probe alignment |
| **M2** Compound | Compound parameter identification (3+ params): Estimate lift-off, conductivity, probe alignment |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate lift-off, conductivity, probe alignment |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate lift-off, conductivity, probe alignment |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct lift-off, conductivity scale |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct lift-off, conductivity scale |
| **M2** Compound | Compound correction with rho measurement: Correct lift-off, conductivity scale |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct lift-off, conductivity scale |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct lift-off, conductivity scale |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
