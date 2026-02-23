# Hyperspectral Remote Sensing (`hyperspectral_remote`)

**Category**: Remote Sensing | **Canonical DAG**: M --> W --> Sigma --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: spectral_unmixing

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Spectral range, spatial resolution, push-broom vs snapshot |
| **M1** Synthetic | Prompt tested with synthetic data validation: Spectral range, spatial resolution, push-broom vs snapshot |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Spectral range, spatial resolution, push-broom vs snapshot |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Spectral range, spatial resolution, push-broom vs snapshot |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Hyperspectral Remote Sensing |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Unmixing under atmospheric correction error, smile/keystone |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Unmixing under atmospheric correction error, smile/keystone |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Unmixing under atmospheric correction error, smile/keystone |
| **M3** Real Data | Real experimental data with measured mismatch: Unmixing under atmospheric correction error, smile/keystone |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Unmixing under atmospheric correction error, smile/keystone |

### Mismatch Parameters
M→W→Sigma→D. Smile [0,2] px, keystone [0,2] px, atmospheric +/-10%.

### Solvers & Expected Performance
- **Solver**: spectral_unmixing

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> W --> Sigma --> D: Estimate smile/keystone, atmospheric parameters |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate smile/keystone, atmospheric parameters |
| **M2** Compound | Compound parameter identification (3+ params): Estimate smile/keystone, atmospheric parameters |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate smile/keystone, atmospheric parameters |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate smile/keystone, atmospheric parameters |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct spectral distortion, atmosphere |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct spectral distortion, atmosphere |
| **M2** Compound | Compound correction with rho measurement: Correct spectral distortion, atmosphere |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct spectral distortion, atmosphere |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct spectral distortion, atmosphere |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
