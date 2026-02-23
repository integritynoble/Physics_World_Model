# Proton Radiography (`proton_radiography`)

**Category**: Scientific Instrumentation | **Canonical DAG**: Pi --> D | **Carrier**: Proton
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: filtered_back_projection

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Beam energy, detector stack, angular acceptance |
| **M1** Synthetic | Prompt tested with synthetic data validation: Beam energy, detector stack, angular acceptance |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Beam energy, detector stack, angular acceptance |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Beam energy, detector stack, angular acceptance |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Proton Radiography |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: MLP recon under MCS model error |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: MLP recon under MCS model error |
| **M2** Compound | Compound mismatch (3+ params simultaneously): MLP recon under MCS model error |
| **M3** Real Data | Real experimental data with measured mismatch: MLP recon under MCS model error |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: MLP recon under MCS model error |

### Mismatch Parameters
Pi→D. MCS error [0,15%], energy loss +/-5%.

### Solvers & Expected Performance
- **Solver**: filtered_back_projection

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for Pi --> D: Estimate scattering model parameters, energy loss |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate scattering model parameters, energy loss |
| **M2** Compound | Compound parameter identification (3+ params): Estimate scattering model parameters, energy loss |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate scattering model parameters, energy loss |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate scattering model parameters, energy loss |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct MCS model, energy calibration |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct MCS model, energy calibration |
| **M2** Compound | Compound correction with rho measurement: Correct MCS model, energy calibration |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct MCS model, energy calibration |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct MCS model, energy calibration |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
