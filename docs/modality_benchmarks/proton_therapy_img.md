# Proton Therapy Imaging (`proton_therapy_img`)

**Category**: Medical Imaging | **Canonical DAG**: Pi --> D | **Carrier**: Proton
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: back_projection

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Energy range, detector stack, range verification |
| **M1** Synthetic | Prompt tested with synthetic data validation: Energy range, detector stack, range verification |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Energy range, detector stack, range verification |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Energy range, detector stack, range verification |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Proton Therapy Imaging |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Backprojection under range uncertainty, scattering |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Backprojection under range uncertainty, scattering |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Backprojection under range uncertainty, scattering |
| **M3** Real Data | Real experimental data with measured mismatch: Backprojection under range uncertainty, scattering |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Backprojection under range uncertainty, scattering |

### Mismatch Parameters
Pi→D, Proton. Range uncertainty +/-3%, MCS error [0,10%].

### Solvers & Expected Performance
- **Solver**: back_projection

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for Pi --> D: Estimate water-equivalent path length, scattering model |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate water-equivalent path length, scattering model |
| **M2** Compound | Compound parameter identification (3+ params): Estimate water-equivalent path length, scattering model |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate water-equivalent path length, scattering model |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate water-equivalent path length, scattering model |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct range model, scattering compensation |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct range model, scattering compensation |
| **M2** Compound | Compound correction with rho measurement: Correct range model, scattering compensation |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct range model, scattering compensation |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct range model, scattering compensation |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
