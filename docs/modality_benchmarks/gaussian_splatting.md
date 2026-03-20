# 3D Gaussian Splatting (3DGS) (`gaussian_splatting`)

**Category**: Neural Rendering | **Canonical DAG**: M --> P --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: gaussian_splatting_3dgs

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Initial point cloud, densification, view selection |
| **M1** Synthetic | Prompt tested with synthetic data validation: Initial point cloud, densification, view selection |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Initial point cloud, densification, view selection |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Initial point cloud, densification, view selection |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for 3D Gaussian Splatting (3DGS) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: 3DGS under SfM initialization error |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: 3DGS under SfM initialization error |
| **M2** Compound | Compound mismatch (3+ params simultaneously): 3DGS under SfM initialization error |
| **M3** Real Data | Real experimental data with measured mismatch: 3DGS under SfM initialization error |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: 3DGS under SfM initialization error |

### Mismatch Parameters
M→P→D. SfM noise [0,0.1], init density [10k,1M], pose error [0,0.03].

### Solvers & Expected Performance
- **Solver**: gaussian_splatting_3dgs

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> P --> D: Estimate point cloud quality, initialization bias |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate point cloud quality, initialization bias |
| **M2** Compound | Compound parameter identification (3+ params): Estimate point cloud quality, initialization bias |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate point cloud quality, initialization bias |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate point cloud quality, initialization bias |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct initialization, re-densify |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct initialization, re-densify |
| **M2** Compound | Compound correction with rho measurement: Correct initialization, re-densify |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct initialization, re-densify |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct initialization, re-densify |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
