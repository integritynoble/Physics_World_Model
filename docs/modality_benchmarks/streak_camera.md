# Streak Camera Imaging (`streak_camera`)

**Category**: Ultrafast Imaging | **Canonical DAG**: M --> Sigma --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: spatiotemporal_recon

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | "Design streak camera system for fluorescence lifetime: 2 ps resolution, 500 ps window." |
| **M1** Synthetic | Prompt tested with synthetic data validation: Sweep speed, slit width, trigger jitter |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Sweep speed, slit width, trigger jitter |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Sweep speed, slit width, trigger jitter |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Streak Camera Imaging |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Spatiotemporal recon under sweep nonlinearity, jitter |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Spatiotemporal recon under sweep nonlinearity, jitter |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Spatiotemporal recon under sweep nonlinearity, jitter |
| **M3** Real Data | Real experimental data with measured mismatch: Spatiotemporal recon under sweep nonlinearity, jitter |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Spatiotemporal recon under sweep nonlinearity, jitter |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Sweep nonlinearity | 0 | [0, 5%] | - |
| Temporal resolution | 1 | [0.5, 5] | ps |
| Dynamic range saturation | 0 | [0, 10%] of pixels | - |
| Trigger jitter | 0 | [0, 10] | ps |

### Solvers & Expected Performance
- **Solver**: spatiotemporal_recon

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> Sigma --> D: Estimate sweep function, trigger jitter, flatfield |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate sweep function, trigger jitter, flatfield |
| **M2** Compound | Compound parameter identification (3+ params): Estimate sweep function, trigger jitter, flatfield |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate sweep function, trigger jitter, flatfield |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate sweep function, trigger jitter, flatfield |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct sweep nonlinearity, jitter compensation |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct sweep nonlinearity, jitter compensation |
| **M2** Compound | Compound correction with rho measurement: Correct sweep nonlinearity, jitter compensation |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct sweep nonlinearity, jitter compensation |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct sweep nonlinearity, jitter compensation |

### Correction Targets
- **Expected rho**: TBD

### Improvement Roadmap
Add synchroscan mode, compressed streak (CUP variant).

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
