# Confocal 3D Z-Stack (`confocal_3d`)

**Category**: Microscopy | **Canonical DAG**: C --> D | **Carrier**: Photon
**Current Maturity**: M1 | **Forward Model**: linear_operator | **Default Solver**: richardson_lucy_3d

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Axial vs lateral resolution, z-step, depth attenuation |
| **M1** Synthetic | Prompt tested with synthetic data validation: Axial vs lateral resolution, z-step, depth attenuation |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Axial vs lateral resolution, z-step, depth attenuation |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Axial vs lateral resolution, z-step, depth attenuation |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Confocal 3D Z-Stack |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: 3D deconvolution under depth-dependent PSF |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: 3D deconvolution under depth-dependent PSF |
| **M2** Compound | Compound mismatch (3+ params simultaneously): 3D deconvolution under depth-dependent PSF |
| **M3** Real Data | Real experimental data with measured mismatch: 3D deconvolution under depth-dependent PSF |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: 3D deconvolution under depth-dependent PSF |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Axial PSF sigma | 3.0 | [1.5, 6.0] | px |
| Refractive index | 1.515 | [1.33, 1.56] | - |
| Attenuation coeff | 0.03 | [0, 0.1] | per slice |
| Spherical aberration | 0 | [0, 0.5] | waves |

### Solvers & Expected Performance
- **Solver**: richardson_lucy_3d

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for C --> D: Estimate 3D PSF, refractive index, attenuation |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate 3D PSF, refractive index, attenuation |
| **M2** Compound | Compound parameter identification (3+ params): Estimate 3D PSF, refractive index, attenuation |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate 3D PSF, refractive index, attenuation |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate 3D PSF, refractive index, attenuation |

### True-Spec Parameters
RI, depth-dependent PSF, attenuation, aberration

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct depth-dependent aberrations |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct depth-dependent aberrations |
| **M2** Compound | Compound correction with rho measurement: Correct depth-dependent aberrations |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct depth-dependent aberrations |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct depth-dependent aberrations |

### Correction Targets
- **Expected rho**: >= 0.70

### Improvement Roadmap
Add depth-dependent PSF model.

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
