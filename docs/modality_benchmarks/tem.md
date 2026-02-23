# Transmission Electron Microscopy (TEM) (`tem`)

**Category**: Electron Microscopy | **Canonical DAG**: C --> D | **Carrier**: Electron
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: ctf_correction

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Acceleration voltage, aperture, defocus series |
| **M1** Synthetic | Prompt tested with synthetic data validation: Acceleration voltage, aperture, defocus series |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Acceleration voltage, aperture, defocus series |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Acceleration voltage, aperture, defocus series |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Transmission Electron Microscopy (TEM) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: CTF correction under defocus, astigmatism, beam tilt |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: CTF correction under defocus, astigmatism, beam tilt |
| **M2** Compound | Compound mismatch (3+ params simultaneously): CTF correction under defocus, astigmatism, beam tilt |
| **M3** Real Data | Real experimental data with measured mismatch: CTF correction under defocus, astigmatism, beam tilt |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: CTF correction under defocus, astigmatism, beam tilt |

### Mismatch Parameters
C→D, Electron. Defocus [-1000,1000] nm, Cs [0.5,2.5] mm, astigmatism [0,100] nm.

### Solvers & Expected Performance
- **Solver**: ctf_correction

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for C --> D: Estimate CTF params (defocus, Cs, astigmatism) |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate CTF params (defocus, Cs, astigmatism) |
| **M2** Compound | Compound parameter identification (3+ params): Estimate CTF params (defocus, Cs, astigmatism) |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate CTF params (defocus, Cs, astigmatism) |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate CTF params (defocus, Cs, astigmatism) |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct CTF, aberration model |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct CTF, aberration model |
| **M2** Compound | Compound correction with rho measurement: Correct CTF, aberration model |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct CTF, aberration model |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct CTF, aberration model |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
