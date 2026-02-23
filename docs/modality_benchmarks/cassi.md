# Coded Aperture Snapshot Spectral Imaging (CASSI) (`cassi`)

**Category**: Compressive Imaging | **Canonical DAG**: M --> W --> Sigma --> D | **Carrier**: Photon
**Current Maturity**: M3 | **Forward Model**: linear_operator | **Default Solver**: mst

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Mask pattern, dispersion element, spectral range, spatial resolution |
| **M1** Synthetic | Prompt tested with synthetic data validation: Mask pattern, dispersion element, spectral range, spatial resolution |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Mask pattern, dispersion element, spectral range, spatial resolution |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Mask pattern, dispersion element, spectral range, spatial resolution |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Coded Aperture Snapshot Spectral Imaging (CASSI) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: GAP-TV / MST-L under mask shift (dx,dy), rotation (theta), dispersion slope (a1), axis offset (alpha) |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: GAP-TV / MST-L under mask shift (dx,dy), rotation (theta), dispersion slope (a1), axis offset (alpha) |
| **M2** Compound | Compound mismatch (3+ params simultaneously): GAP-TV / MST-L under mask shift (dx,dy), rotation (theta), dispersion slope (a1), axis offset (alpha) |
| **M3** Real Data | Real experimental data with measured mismatch: GAP-TV / MST-L under mask shift (dx,dy), rotation (theta), dispersion slope (a1), axis offset (alpha) |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: GAP-TV / MST-L under mask shift (dx,dy), rotation (theta), dispersion slope (a1), axis offset (alpha) |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | True Example | Unit |
|-----------|---------|----------------|--------------|------|
| Mask shift dx | 0 | [-3.0, 3.0] | 1.47 | px |
| Mask shift dy | 0 | [-3.0, 3.0] | -0.23 | px |
| Mask rotation | 0 | [-2.0, 2.0] | 0.31 | deg |
| Dispersion slope a1 | 2.0 | [1.5, 2.5] | 2.01 | px/band |
| Dispersion offset alpha | 0 | [-0.5, 0.5] | 0.04 | px |
| Gain | 1.0 | [0.9, 1.1] | 1.02 | - |
| Read noise | 5.0 | [1.0, 15.0] | 5.1 | e- |

### Solvers & Expected Performance
- **Solver**: mst
- **Validated baseline**: GAP-TV +0.76 dB, rho = 0.85

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> W --> Sigma --> D: Estimate 5 mismatch params from measurement residuals |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate 5 mismatch params from measurement residuals |
| **M2** Compound | Compound parameter identification (3+ params): Estimate 5 mismatch params from measurement residuals |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate 5 mismatch params from measurement residuals |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate 5 mismatch params from measurement residuals |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct all 5 params; rho validated at 85% (flagship) |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct all 5 params; rho validated at 85% (flagship) |
| **M2** Compound | Compound correction with rho measurement: Correct all 5 params; rho validated at 85% (flagship) |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct all 5 params; rho validated at 85% (flagship) |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct all 5 params; rho validated at 85% (flagship) |

### Correction Targets
- **Expected rho**: >= 0.85

### Improvement Roadmap
Compound all 5 params; Red Team adversarial; PSF spectral variation.

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
