# LiDAR Scanner (`lidar`)

**Category**: Depth Imaging | **Canonical DAG**: P --> S --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: point_cloud_recon

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Scan pattern, pulse rate, wavelength, range |
| **M1** Synthetic | Prompt tested with synthetic data validation: Scan pattern, pulse rate, wavelength, range |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Scan pattern, pulse rate, wavelength, range |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Scan pattern, pulse rate, wavelength, range |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for LiDAR Scanner |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Point cloud recon under timing jitter, angular error |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Point cloud recon under timing jitter, angular error |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Point cloud recon under timing jitter, angular error |
| **M3** Real Data | Real experimental data with measured mismatch: Point cloud recon under timing jitter, angular error |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Point cloud recon under timing jitter, angular error |

### Mismatch Parameters
P→S→D. Timing jitter [0,0.5] ns, angular error [-0.1,0.1] deg.

### Solvers & Expected Performance
- **Solver**: point_cloud_recon

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for P --> S --> D: Estimate timing calibration, angular encoder error |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate timing calibration, angular encoder error |
| **M2** Compound | Compound parameter identification (3+ params): Estimate timing calibration, angular encoder error |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate timing calibration, angular encoder error |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate timing calibration, angular encoder error |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct timing, angular calibration |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct timing, angular calibration |
| **M2** Compound | Compound correction with rho measurement: Correct timing, angular calibration |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct timing, angular calibration |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct timing, angular calibration |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
