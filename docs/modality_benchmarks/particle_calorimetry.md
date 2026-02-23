# Particle Calorimetry (`particle_calorimetry`)

**Category**: Broader Experimental Science | **Canonical DAG**: R --> Sigma --> D | **Carrier**: Particle
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: energy_recon

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Absorber/scintillator layers, granularity |
| **M1** Synthetic | Prompt tested with synthetic data validation: Absorber/scintillator layers, granularity |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Absorber/scintillator layers, granularity |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Absorber/scintillator layers, granularity |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Particle Calorimetry |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Energy/position recon under inter-calibration error |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Energy/position recon under inter-calibration error |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Energy/position recon under inter-calibration error |
| **M3** Real Data | Real experimental data with measured mismatch: Energy/position recon under inter-calibration error |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Energy/position recon under inter-calibration error |

### Mismatch Parameters
R→Sigma→D. Inter-cal [0,3%], non-linearity [0,5%].

### Solvers & Expected Performance
- **Solver**: energy_recon

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for R --> Sigma --> D: Estimate cell calibration, non-linearity curve |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate cell calibration, non-linearity curve |
| **M2** Compound | Compound parameter identification (3+ params): Estimate cell calibration, non-linearity curve |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate cell calibration, non-linearity curve |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate cell calibration, non-linearity curve |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct inter-calibration, non-linearity |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct inter-calibration, non-linearity |
| **M2** Compound | Compound correction with rho measurement: Correct inter-calibration, non-linearity |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct inter-calibration, non-linearity |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct inter-calibration, non-linearity |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
