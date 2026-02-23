# Lensless (Diffuser Camera) Imaging (`lensless`)

**Category**: Computational Photography | **Canonical DAG**: C --> D | **Carrier**: Photon
**Current Maturity**: M3 | **Forward Model**: linear_operator | **Default Solver**: admm_tv

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Diffuser/mask type, sensor distance, PSF calibration |
| **M1** Synthetic | Prompt tested with synthetic data validation: Diffuser/mask type, sensor distance, PSF calibration |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Diffuser/mask type, sensor distance, PSF calibration |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Diffuser/mask type, sensor distance, PSF calibration |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Lensless (Diffuser Camera) Imaging |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: ADMM under PSF shift, scale drift, defocus |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: ADMM under PSF shift, scale drift, defocus |
| **M2** Compound | Compound mismatch (3+ params simultaneously): ADMM under PSF shift, scale drift, defocus |
| **M3** Real Data | Real experimental data with measured mismatch: ADMM under PSF shift, scale drift, defocus |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: ADMM under PSF shift, scale drift, defocus |

### Mismatch Parameters
PSF shift [−5,5] px, scale [0.9,1.1], defocus +/−50 um, rotation [−2,2] deg.

### Solvers & Expected Performance
- **Solver**: admm_tv
- **Validated baseline**: ADMM +3.55 dB, rho = 0.78

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for C --> D: Estimate PSF shift, scale, defocus offset |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate PSF shift, scale, defocus offset |
| **M2** Compound | Compound parameter identification (3+ params): Estimate PSF shift, scale, defocus offset |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate PSF shift, scale, defocus offset |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate PSF shift, scale, defocus offset |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct PSF model; rho=78%, +3.55 dB |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct PSF model; rho=78%, +3.55 dB |
| **M2** Compound | Compound correction with rho measurement: Correct PSF model; rho=78%, +3.55 dB |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct PSF model; rho=78%, +3.55 dB |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct PSF model; rho=78%, +3.55 dB |

### Correction Targets
- **Expected rho**: 0.78

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
