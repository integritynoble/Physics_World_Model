# High Dynamic Range (HDR) Imaging (`hdr_imaging`)

**Category**: Computational Photography | **Canonical DAG**: M --> Sigma --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: hdr_merge

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Exposure bracketing, tone mapping, dynamic range |
| **M1** Synthetic | Prompt tested with synthetic data validation: Exposure bracketing, tone mapping, dynamic range |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Exposure bracketing, tone mapping, dynamic range |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Exposure bracketing, tone mapping, dynamic range |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for High Dynamic Range (HDR) Imaging |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: HDR merge under exposure time error, ghosting |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: HDR merge under exposure time error, ghosting |
| **M2** Compound | Compound mismatch (3+ params simultaneously): HDR merge under exposure time error, ghosting |
| **M3** Real Data | Real experimental data with measured mismatch: HDR merge under exposure time error, ghosting |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: HDR merge under exposure time error, ghosting |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Camera response function error | 0 | [0, 10%] | - |
| Exposure ratio error | 0 | [-10%, 10%] | - |
| Ghost artifact (motion between exposures) | 0 | [0, 5] | px |

### Solvers & Expected Performance
- **Solver**: hdr_merge

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> Sigma --> D: Estimate exposure ratios, CRF, motion between frames |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate exposure ratios, CRF, motion between frames |
| **M2** Compound | Compound parameter identification (3+ params): Estimate exposure ratios, CRF, motion between frames |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate exposure ratios, CRF, motion between frames |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate exposure ratios, CRF, motion between frames |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct exposure calibration, deghosting |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct exposure calibration, deghosting |
| **M2** Compound | Compound correction with rho measurement: Correct exposure calibration, deghosting |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct exposure calibration, deghosting |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct exposure calibration, deghosting |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
