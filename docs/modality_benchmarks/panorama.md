# Panorama Multi-Focus Fusion (`panorama`)

**Category**: Computational Photography | **Canonical DAG**: C --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: laplacian_pyramid_fusion

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Focal sweep range, focal planes, depth of field |
| **M1** Synthetic | Prompt tested with synthetic data validation: Focal sweep range, focal planes, depth of field |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Focal sweep range, focal planes, depth of field |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Focal sweep range, focal planes, depth of field |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Panorama Multi-Focus Fusion |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Laplacian pyramid under focus distance error |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Laplacian pyramid under focus distance error |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Laplacian pyramid under focus distance error |
| **M3** Real Data | Real experimental data with measured mismatch: Laplacian pyramid under focus distance error |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Laplacian pyramid under focus distance error |

### Mismatch Parameters
Focus distance +/−10%, registration [0,3] px.

### Solvers & Expected Performance
- **Solver**: laplacian_pyramid_fusion

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for C --> D: Estimate focal distances, aperture, depth map |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate focal distances, aperture, depth map |
| **M2** Compound | Compound parameter identification (3+ params): Estimate focal distances, aperture, depth map |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate focal distances, aperture, depth map |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate focal distances, aperture, depth map |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct focal plane registration |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct focal plane registration |
| **M2** Compound | Compound correction with rho measurement: Correct focal plane registration |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct focal plane registration |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct focal plane registration |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
