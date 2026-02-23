# Atom Probe Tomography (APT) (`atom_probe`)

**Category**: Scientific Instrumentation | **Canonical DAG**: S --> D | **Carrier**: Ion
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: trajectory_recon

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Voltage/laser pulse, detection efficiency, FOV |
| **M1** Synthetic | Prompt tested with synthetic data validation: Voltage/laser pulse, detection efficiency, FOV |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Voltage/laser pulse, detection efficiency, FOV |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Voltage/laser pulse, detection efficiency, FOV |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Atom Probe Tomography (APT) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: 3D recon under trajectory aberration, local magnification |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: 3D recon under trajectory aberration, local magnification |
| **M2** Compound | Compound mismatch (3+ params simultaneously): 3D recon under trajectory aberration, local magnification |
| **M3** Real Data | Real experimental data with measured mismatch: 3D recon under trajectory aberration, local magnification |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: 3D recon under trajectory aberration, local magnification |

### Mismatch Parameters
S→D. Trajectory aberration [0,10%], local magnification [0,20%].

### Solvers & Expected Performance
- **Solver**: trajectory_recon

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for S --> D: Estimate tip shape, local magnification |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate tip shape, local magnification |
| **M2** Compound | Compound parameter identification (3+ params): Estimate tip shape, local magnification |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate tip shape, local magnification |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate tip shape, local magnification |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct geometry, compositional bias |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct geometry, compositional bias |
| **M2** Compound | Compound correction with rho measurement: Correct geometry, compositional bias |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct geometry, compositional bias |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct geometry, compositional bias |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
