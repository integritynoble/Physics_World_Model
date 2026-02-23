# Flash LiDAR (`flash_lidar`)

**Category**: Depth Imaging | **Canonical DAG**: P --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: depth_map_recon

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Laser power, APD array, range gate |
| **M1** Synthetic | Prompt tested with synthetic data validation: Laser power, APD array, range gate |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Laser power, APD array, range gate |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Laser power, APD array, range gate |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Flash LiDAR |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Depth map under background noise, multi-return |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Depth map under background noise, multi-return |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Depth map under background noise, multi-return |
| **M3** Real Data | Real experimental data with measured mismatch: Depth map under background noise, multi-return |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Depth map under background noise, multi-return |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| SPAD jitter | 0 | [0, 100] | ps |
| Ambient photon rate | 0 | [0, 10x signal] | - |
| Pile-up distortion | 0 | [0, 20%] | at high flux |
| Pixel cross-talk | 0 | [0, 5%] | - |

### Solvers & Expected Performance
- **Solver**: depth_map_recon

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for P --> D: Estimate background rate, reflectivity, range bias |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate background rate, reflectivity, range bias |
| **M2** Compound | Compound parameter identification (3+ params): Estimate background rate, reflectivity, range bias |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate background rate, reflectivity, range bias |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate background rate, reflectivity, range bias |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct range calibration, background subtraction |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct range calibration, background subtraction |
| **M2** Compound | Compound correction with rho measurement: Correct range calibration, background subtraction |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct range calibration, background subtraction |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct range calibration, background subtraction |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
