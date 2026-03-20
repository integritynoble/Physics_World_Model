# Diffusion MRI (DTI) (`diffusion_mri`)

**Category**: Medical Imaging | **Canonical DAG**: M --> F --> S --> D | **Carrier**: Spin/RF
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: weighted_least_squares

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: b-values, gradient directions, eddy currents |
| **M1** Synthetic | Prompt tested with synthetic data validation: b-values, gradient directions, eddy currents |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for b-values, gradient directions, eddy currents |
| **M3** Real Data | Grounded in real experimental/clinical protocols: b-values, gradient directions, eddy currents |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Diffusion MRI (DTI) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: WLS tensor fitting under gradient nonlinearity, eddy current distortion |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: WLS tensor fitting under gradient nonlinearity, eddy current distortion |
| **M2** Compound | Compound mismatch (3+ params simultaneously): WLS tensor fitting under gradient nonlinearity, eddy current distortion |
| **M3** Real Data | Real experimental data with measured mismatch: WLS tensor fitting under gradient nonlinearity, eddy current distortion |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: WLS tensor fitting under gradient nonlinearity, eddy current distortion |

### Mismatch Parameters
M→F→S→D, Spin/RF. Gradient nonlinearity [0,5%], eddy distortion [0,3] px.

### Solvers & Expected Performance
- **Solver**: weighted_least_squares

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> F --> S --> D: Estimate gradient tables, eddy current coefficients |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate gradient tables, eddy current coefficients |
| **M2** Compound | Compound parameter identification (3+ params): Estimate gradient tables, eddy current coefficients |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate gradient tables, eddy current coefficients |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate gradient tables, eddy current coefficients |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct gradient nonlinearity, eddy currents |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct gradient nonlinearity, eddy currents |
| **M2** Compound | Compound correction with rho measurement: Correct gradient nonlinearity, eddy currents |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct gradient nonlinearity, eddy currents |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct gradient nonlinearity, eddy currents |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
