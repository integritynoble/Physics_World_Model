# SPECT/CT Fusion (`spect_ct`)

**Category**: Multi-Modal Fusion | **Canonical DAG**: Pi --> D (SPECT) + Pi --> D (CT) --> Fusion | **Carrier**: Gamma + X-ray
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: joint_recon_spectct

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Collimator-CT geometry, energy windows, attenuation |
| **M1** Synthetic | Prompt tested with synthetic data validation: Collimator-CT geometry, energy windows, attenuation |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Collimator-CT geometry, energy windows, attenuation |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Collimator-CT geometry, energy windows, attenuation |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for SPECT/CT Fusion |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Joint recon under SPECT-CT misregistration |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Joint recon under SPECT-CT misregistration |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Joint recon under SPECT-CT misregistration |
| **M3** Real Data | Real experimental data with measured mismatch: Joint recon under SPECT-CT misregistration |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Joint recon under SPECT-CT misregistration |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Registration error | 0 | [0, 5] | mm |
| CT-based attenuation error | 0 | [0, 10%] | - |
| Scatter correction error | 0 | [0, 15%] | - |

### Solvers & Expected Performance
- **Solver**: joint_recon_spectct

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for Pi --> D (SPECT) + Pi --> D (CT) --> Fusion: Estimate registration offset, CT HU to mu conversion |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate registration offset, CT HU to mu conversion |
| **M2** Compound | Compound parameter identification (3+ params): Estimate registration offset, CT HU to mu conversion |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate registration offset, CT HU to mu conversion |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate registration offset, CT HU to mu conversion |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct co-registration, attenuation |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct co-registration, attenuation |
| **M2** Compound | Compound correction with rho measurement: Correct co-registration, attenuation |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct co-registration, attenuation |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct co-registration, attenuation |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
