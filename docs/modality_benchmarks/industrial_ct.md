# Industrial X-ray CT (`industrial_ct`)

**Category**: Industrial Inspection | **Canonical DAG**: Pi --> D | **Carrier**: X-ray
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: fbp

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: kV/mA, geometry, magnification, voxel size |
| **M1** Synthetic | Prompt tested with synthetic data validation: kV/mA, geometry, magnification, voxel size |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for kV/mA, geometry, magnification, voxel size |
| **M3** Real Data | Grounded in real experimental/clinical protocols: kV/mA, geometry, magnification, voxel size |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Industrial X-ray CT |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: FBP/iterative under scatter, beam hardening, ring artifacts |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: FBP/iterative under scatter, beam hardening, ring artifacts |
| **M2** Compound | Compound mismatch (3+ params simultaneously): FBP/iterative under scatter, beam hardening, ring artifacts |
| **M3** Real Data | Real experimental data with measured mismatch: FBP/iterative under scatter, beam hardening, ring artifacts |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: FBP/iterative under scatter, beam hardening, ring artifacts |

### Mismatch Parameters
Pi→D. CoR [-5,5] px, scatter [0.1,0.6], beam hardening, ring artifacts.

### Solvers & Expected Performance
- **Solver**: fbp

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for Pi --> D: Estimate center offset, ring sources, scatter fraction |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate center offset, ring sources, scatter fraction |
| **M2** Compound | Compound parameter identification (3+ params): Estimate center offset, ring sources, scatter fraction |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate center offset, ring sources, scatter fraction |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate center offset, ring sources, scatter fraction |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct geometry, scatter, beam hardening |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct geometry, scatter, beam hardening |
| **M2** Compound | Compound correction with rho measurement: Correct geometry, scatter, beam hardening |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct geometry, scatter, beam hardening |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct geometry, scatter, beam hardening |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
