# 4D-STEM Electron Diffraction (`electron_diffraction`)

**Category**: Electron Microscopy | **Canonical DAG**: M --> P --> D | **Carrier**: Electron
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: ptychography_epie

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Probe size, scan step, camera length |
| **M1** Synthetic | Prompt tested with synthetic data validation: Probe size, scan step, camera length |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Probe size, scan step, camera length |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Probe size, scan step, camera length |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for 4D-STEM Electron Diffraction |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Ptychographic recon under camera length error |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Ptychographic recon under camera length error |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Ptychographic recon under camera length error |
| **M3** Real Data | Real experimental data with measured mismatch: Ptychographic recon under camera length error |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Ptychographic recon under camera length error |

### Mismatch Parameters
M→P→D, Electron. Camera length +/-5%, beam center +/-5 px.

### Solvers & Expected Performance
- **Solver**: ptychography_epie

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> P --> D: Estimate camera length, beam center, rotation |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate camera length, beam center, rotation |
| **M2** Compound | Compound parameter identification (3+ params): Estimate camera length, beam center, rotation |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate camera length, beam center, rotation |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate camera length, beam center, rotation |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct geometry calibration |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct geometry calibration |
| **M2** Compound | Compound correction with rho measurement: Correct geometry calibration |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct geometry calibration |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct geometry calibration |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
