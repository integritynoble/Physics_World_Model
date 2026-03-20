# PET/MR Fusion (`pet_mr`)

**Category**: Multi-Modal Fusion | **Canonical DAG**: Pi --> D (PET) + M --> F --> S --> D (MR) --> Fusion | **Carrier**: Gamma + RF
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: joint_recon_petmr

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Simultaneous vs sequential, attenuation from MR |
| **M1** Synthetic | Prompt tested with synthetic data validation: Simultaneous vs sequential, attenuation from MR |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Simultaneous vs sequential, attenuation from MR |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Simultaneous vs sequential, attenuation from MR |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for PET/MR Fusion |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Joint recon under MR-based attenuation error, susceptibility |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Joint recon under MR-based attenuation error, susceptibility |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Joint recon under MR-based attenuation error, susceptibility |
| **M3** Real Data | Real experimental data with measured mismatch: Joint recon under MR-based attenuation error, susceptibility |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Joint recon under MR-based attenuation error, susceptibility |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| MR-based attenuation error | 0 | [0, 15%] | - |
| Susceptibility artifact at air/tissue | 0 | [0, 5] | mm |
| Timing synchronization | 0 | [0, 100] | ms |
| Truncation (MR FOV < PET FOV) | 0 | [0, 20%] | of body |

### Solvers & Expected Performance
- **Solver**: joint_recon_petmr

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for Pi --> D (PET) + M --> F --> S --> D (MR) --> Fusion: Estimate attenuation map, geometric distortion |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate attenuation map, geometric distortion |
| **M2** Compound | Compound parameter identification (3+ params): Estimate attenuation map, geometric distortion |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate attenuation map, geometric distortion |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate attenuation map, geometric distortion |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct MR-AC, reduce susceptibility artifacts |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct MR-AC, reduce susceptibility artifacts |
| **M2** Compound | Compound correction with rho measurement: Correct MR-AC, reduce susceptibility artifacts |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct MR-AC, reduce susceptibility artifacts |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct MR-AC, reduce susceptibility artifacts |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
