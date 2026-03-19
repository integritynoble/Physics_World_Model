# Active Thermography (IR) (`active_thermography`)

**Category**: Industrial Inspection | **Canonical DAG**: P --> D | **Carrier**: IR photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: thermal_diffusivity_inversion

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Excitation type, camera NETD, frame rate |
| **M1** Synthetic | Prompt tested with synthetic data validation: Excitation type, camera NETD, frame rate |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Excitation type, camera NETD, frame rate |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Excitation type, camera NETD, frame rate |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Active Thermography (IR) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Thermal diffusivity inversion under non-uniform heating |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Thermal diffusivity inversion under non-uniform heating |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Thermal diffusivity inversion under non-uniform heating |
| **M3** Real Data | Real experimental data with measured mismatch: Thermal diffusivity inversion under non-uniform heating |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Thermal diffusivity inversion under non-uniform heating |

### Mismatch Parameters
P→D. Emissivity [0,15%], heating uniformity [0.7,1.3].

### Solvers & Expected Performance
- **Solver**: thermal_diffusivity_inversion

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for P --> D: Estimate emissivity map, heating uniformity |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate emissivity map, heating uniformity |
| **M2** Compound | Compound parameter identification (3+ params): Estimate emissivity map, heating uniformity |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate emissivity map, heating uniformity |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate emissivity map, heating uniformity |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct emissivity, excitation model |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct emissivity, excitation model |
| **M2** Compound | Compound correction with rho measurement: Correct emissivity, excitation model |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct emissivity, excitation model |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct emissivity, excitation model |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
