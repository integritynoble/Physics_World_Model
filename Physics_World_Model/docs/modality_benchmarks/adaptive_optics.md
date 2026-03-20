# Adaptive Optics (AO) Imaging (`adaptive_optics`)

**Category**: Broader Experimental Science | **Canonical DAG**: M --> C --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: psf_deconvolution

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Wavefront sensor, DM actuators, guide star |
| **M1** Synthetic | Prompt tested with synthetic data validation: Wavefront sensor, DM actuators, guide star |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Wavefront sensor, DM actuators, guide star |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Wavefront sensor, DM actuators, guide star |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Adaptive Optics (AO) Imaging |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: PSF recon under residual wavefront error |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: PSF recon under residual wavefront error |
| **M2** Compound | Compound mismatch (3+ params simultaneously): PSF recon under residual wavefront error |
| **M3** Real Data | Real experimental data with measured mismatch: PSF recon under residual wavefront error |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: PSF recon under residual wavefront error |

### Mismatch Parameters
M→C→D. Residual wavefront [0,lambda/4], r0 [5,30] cm, wind [5,30] m/s.

### Solvers & Expected Performance
- **Solver**: psf_deconvolution

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> C --> D: Estimate residual wavefront, Cn2 profile |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate residual wavefront, Cn2 profile |
| **M2** Compound | Compound parameter identification (3+ params): Estimate residual wavefront, Cn2 profile |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate residual wavefront, Cn2 profile |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate residual wavefront, Cn2 profile |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct AO loop, turbulence model |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct AO loop, turbulence model |
| **M2** Compound | Compound correction with rho measurement: Correct AO loop, turbulence model |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct AO loop, turbulence model |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct AO loop, turbulence model |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
