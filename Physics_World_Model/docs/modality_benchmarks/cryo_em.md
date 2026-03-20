# Cryo-EM Single Particle Analysis (`cryo_em`)

**Category**: Scientific Instrumentation | **Canonical DAG**: C --> D | **Carrier**: Electron
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: ctf_3d_refinement

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Voltage, defocus range, dose, ice thickness |
| **M1** Synthetic | Prompt tested with synthetic data validation: Voltage, defocus range, dose, ice thickness |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Voltage, defocus range, dose, ice thickness |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Voltage, defocus range, dose, ice thickness |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Cryo-EM Single Particle Analysis |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: CTF correction + 3D refinement under beam tilt |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: CTF correction + 3D refinement under beam tilt |
| **M2** Compound | Compound mismatch (3+ params simultaneously): CTF correction + 3D refinement under beam tilt |
| **M3** Real Data | Real experimental data with measured mismatch: CTF correction + 3D refinement under beam tilt |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: CTF correction + 3D refinement under beam tilt |

### Mismatch Parameters
C→D. Defocus [-0.5,-5.0] um, Cs [2.0,3.5] mm, beam tilt [0,1] mrad, ice [20,100] nm. rho >= 0.85.

### Solvers & Expected Performance
- **Solver**: ctf_3d_refinement

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for C --> D: Estimate CTF per micrograph, beam tilt, ice |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate CTF per micrograph, beam tilt, ice |
| **M2** Compound | Compound parameter identification (3+ params): Estimate CTF per micrograph, beam tilt, ice |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate CTF per micrograph, beam tilt, ice |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate CTF per micrograph, beam tilt, ice |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct CTF, beam tilt, Ewald sphere |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct CTF, beam tilt, Ewald sphere |
| **M2** Compound | Compound correction with rho measurement: Correct CTF, beam tilt, Ewald sphere |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct CTF, beam tilt, Ewald sphere |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct CTF, beam tilt, Ewald sphere |

### Correction Targets
- **Expected rho**: >= 0.85.

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
