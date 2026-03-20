# X-ray Angiography (`angiography`)

**Category**: Medical Imaging | **Canonical DAG**: Pi --> D | **Carrier**: X-ray
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: dsa_subtraction

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Contrast timing, frame rate, subtraction protocol |
| **M1** Synthetic | Prompt tested with synthetic data validation: Contrast timing, frame rate, subtraction protocol |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Contrast timing, frame rate, subtraction protocol |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Contrast timing, frame rate, subtraction protocol |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for X-ray Angiography |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: DSA subtraction under patient motion, misregistration |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: DSA subtraction under patient motion, misregistration |
| **M2** Compound | Compound mismatch (3+ params simultaneously): DSA subtraction under patient motion, misregistration |
| **M3** Real Data | Real experimental data with measured mismatch: DSA subtraction under patient motion, misregistration |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: DSA subtraction under patient motion, misregistration |

### Mismatch Parameters
Pi → D, X-ray. Motion [0,10] px, misregistration [-3,3] deg.

### Solvers & Expected Performance
- **Solver**: dsa_subtraction

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for Pi --> D: Estimate motion field between mask and contrast frames |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate motion field between mask and contrast frames |
| **M2** Compound | Compound parameter identification (3+ params): Estimate motion field between mask and contrast frames |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate motion field between mask and contrast frames |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate motion field between mask and contrast frames |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct motion-compensated subtraction |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct motion-compensated subtraction |
| **M2** Compound | Compound correction with rho measurement: Correct motion-compensated subtraction |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct motion-compensated subtraction |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct motion-compensated subtraction |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
