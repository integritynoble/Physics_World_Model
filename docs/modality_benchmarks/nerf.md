# Neural Radiance Fields (NeRF) (`nerf`)

**Category**: Neural Rendering | **Canonical DAG**: M --> P --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: nerf_mlp

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: View count, camera placement, scene bounds |
| **M1** Synthetic | Prompt tested with synthetic data validation: View count, camera placement, scene bounds |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for View count, camera placement, scene bounds |
| **M3** Real Data | Grounded in real experimental/clinical protocols: View count, camera placement, scene bounds |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Neural Radiance Fields (NeRF) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: NeRF/Instant-NGP under camera pose error, intrinsic error |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: NeRF/Instant-NGP under camera pose error, intrinsic error |
| **M2** Compound | Compound mismatch (3+ params simultaneously): NeRF/Instant-NGP under camera pose error, intrinsic error |
| **M3** Real Data | Real experimental data with measured mismatch: NeRF/Instant-NGP under camera pose error, intrinsic error |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: NeRF/Instant-NGP under camera pose error, intrinsic error |

### Mismatch Parameters
M→P→D. Pose error [0,0.05] scene units, rotation [0,3] deg, focal +/-5%. rho >= 0.80.

### Solvers & Expected Performance
- **Solver**: nerf_mlp

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> P --> D: Estimate camera poses, focal length, distortion |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate camera poses, focal length, distortion |
| **M2** Compound | Compound parameter identification (3+ params): Estimate camera poses, focal length, distortion |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate camera poses, focal length, distortion |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate camera poses, focal length, distortion |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct camera calibration, refine poses |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct camera calibration, refine poses |
| **M2** Compound | Compound correction with rho measurement: Correct camera calibration, refine poses |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct camera calibration, refine poses |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct camera calibration, refine poses |

### Correction Targets
- **Expected rho**: >= 0.80

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
