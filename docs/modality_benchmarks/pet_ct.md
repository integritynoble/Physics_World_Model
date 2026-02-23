# PET/CT Fusion (`pet_ct`)

**Category**: Multi-Modal Fusion | **Canonical DAG**: Pi --> D (CT) + Pi --> D (PET) --> Fusion | **Carrier**: X-ray + Gamma
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: joint_recon_petct

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | "Design PET/CT protocol for lung staging: low-dose CT, FDG-PET, 3-min beds, gated." |
| **M1** Synthetic | Prompt tested with synthetic data validation: PET/CT geometry, gantry offset, attenuation protocol |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for PET/CT geometry, gantry offset, attenuation protocol |
| **M3** Real Data | Grounded in real experimental/clinical protocols: PET/CT geometry, gantry offset, attenuation protocol |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for PET/CT Fusion |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Joint recon under PET-CT misregistration, attenuation error |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Joint recon under PET-CT misregistration, attenuation error |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Joint recon under PET-CT misregistration, attenuation error |
| **M3** Real Data | Real experimental data with measured mismatch: Joint recon under PET-CT misregistration, attenuation error |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Joint recon under PET-CT misregistration, attenuation error |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| CT-PET registration error | 0 | [0, 3] | mm |
| Attenuation map from CT error | 0 | [0, 10%] | HU-to-LAC |
| Respiratory motion mismatch | 0 | [0, 15] | mm |
| CT contrast agent artifact | 0 | [0, 20%] | attenuation |

### Solvers & Expected Performance
- **Solver**: joint_recon_petct

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for Pi --> D (CT) + Pi --> D (PET) --> Fusion: Estimate spatial misregistration, attenuation map error |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate spatial misregistration, attenuation map error |
| **M2** Compound | Compound parameter identification (3+ params): Estimate spatial misregistration, attenuation map error |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate spatial misregistration, attenuation map error |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate spatial misregistration, attenuation map error |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct co-registration, attenuation map |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct co-registration, attenuation map |
| **M2** Compound | Compound correction with rho measurement: Correct co-registration, attenuation map |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct co-registration, attenuation map |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct co-registration, attenuation map |

### Correction Targets
- **Expected rho**: TBD

### Improvement Roadmap
Respiratory gating, metal artifact propagation to PET.

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
