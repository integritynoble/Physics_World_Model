# Fourier Ptychographic Microscopy (FPM) (`fpm`)

**Category**: Microscopy | **Canonical DAG**: M --> P --> D | **Carrier**: Photon
**Current Maturity**: M1 | **Forward Model**: nonlinear_operator | **Default Solver**: sequential_phase_retrieval

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: LED array geometry, overlap ratio, NA synthesis |
| **M1** Synthetic | Prompt tested with synthetic data validation: LED array geometry, overlap ratio, NA synthesis |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for LED array geometry, overlap ratio, NA synthesis |
| **M3** Real Data | Grounded in real experimental/clinical protocols: LED array geometry, overlap ratio, NA synthesis |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Fourier Ptychographic Microscopy (FPM) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Phase retrieval under LED position error |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Phase retrieval under LED position error |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Phase retrieval under LED position error |
| **M3** Real Data | Real experimental data with measured mismatch: Phase retrieval under LED position error |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Phase retrieval under LED position error |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| LED position error | 0 | +/- 0.5 mm each | mm |
| LED intensity variation | 1.0 | [0.5, 1.5] per LED | relative |
| Pupil aberration (Zernike) | 0 | [0, 0.3] waves/mode | waves |
| Defocus | 0 | [-5, 5] | um |

### Solvers & Expected Performance
- **Solver**: sequential_phase_retrieval

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> P --> D: Estimate LED positions, aberrations, intensity |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate LED positions, aberrations, intensity |
| **M2** Compound | Compound parameter identification (3+ params): Estimate LED positions, aberrations, intensity |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate LED positions, aberrations, intensity |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate LED positions, aberrations, intensity |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct LED misalignment, pupil aberration |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct LED misalignment, pupil aberration |
| **M2** Compound | Compound correction with rho measurement: Correct LED misalignment, pupil aberration |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct LED misalignment, pupil aberration |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct LED misalignment, pupil aberration |

### Correction Targets
- **Expected rho**: >= 0.85

### Improvement Roadmap
Add vignetting, LED failure robustness.

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
