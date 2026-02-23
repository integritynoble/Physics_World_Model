# CT + Fluorescence (FLIT) (`ct_fluorescence`)

**Category**: Multi-Modal Fusion | **Canonical DAG**: Pi --> D (CT) + M --> R,P --> D (FLI) --> Fusion | **Carrier**: X-ray + Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: joint_recon_flit

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: CT structural prior, fluorophore, optical model |
| **M1** Synthetic | Prompt tested with synthetic data validation: CT structural prior, fluorophore, optical model |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for CT structural prior, fluorophore, optical model |
| **M3** Real Data | Grounded in real experimental/clinical protocols: CT structural prior, fluorophore, optical model |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for CT + Fluorescence (FLIT) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Joint recon under fluence error, tissue heterogeneity |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Joint recon under fluence error, tissue heterogeneity |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Joint recon under fluence error, tissue heterogeneity |
| **M3** Real Data | Real experimental data with measured mismatch: Joint recon under fluence error, tissue heterogeneity |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Joint recon under fluence error, tissue heterogeneity |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Optical property assignment error | 0 | [0, 30%] | - |
| Autofluorescence | 0 | [0, 50%] of signal | - |
| Registration (CT to optical) | 0 | [0, 3] | mm |

### Solvers & Expected Performance
- **Solver**: joint_recon_flit

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for Pi --> D (CT) + M --> R,P --> D (FLI) --> Fusion: Estimate fluorophore distribution, optical properties |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate fluorophore distribution, optical properties |
| **M2** Compound | Compound parameter identification (3+ params): Estimate fluorophore distribution, optical properties |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate fluorophore distribution, optical properties |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate fluorophore distribution, optical properties |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct fluence, structural prior alignment |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct fluence, structural prior alignment |
| **M2** Compound | Compound correction with rho measurement: Correct fluence, structural prior alignment |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct fluence, structural prior alignment |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct fluence, structural prior alignment |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
