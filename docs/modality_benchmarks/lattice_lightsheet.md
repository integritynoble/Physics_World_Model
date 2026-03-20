# Lattice Light-Sheet Microscopy (`lattice_lightsheet`)

**Category**: Microscopy | **Canonical DAG**: C --> D | **Carrier**: Photon
**Current Maturity**: M0 | **Forward Model**: linear_operator | **Default Solver**: richardson_lucy_3d

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Lattice pattern, dithering, z-step, tile overlap |
| **M1** Synthetic | Prompt tested with synthetic data validation: Lattice pattern, dithering, z-step, tile overlap |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Lattice pattern, dithering, z-step, tile overlap |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Lattice pattern, dithering, z-step, tile overlap |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Lattice Light-Sheet Microscopy |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Deconvolution + tile stitching under lattice error |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Deconvolution + tile stitching under lattice error |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Deconvolution + tile stitching under lattice error |
| **M3** Real Data | Real experimental data with measured mismatch: Deconvolution + tile stitching under lattice error |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Deconvolution + tile stitching under lattice error |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Lattice period error | 0 | [-5%, 5%] | relative |
| Dithering range | correct | +/- 10% | - |
| Sheet NA error | 0 | [-0.05, 0.05] | - |
| Excitation PSF sidelobe | 0 | [0, 10%] | relative |

### Solvers & Expected Performance
- **Solver**: richardson_lucy_3d

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for C --> D: Estimate lattice period, dither range, tile offset |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate lattice period, dither range, tile offset |
| **M2** Compound | Compound parameter identification (3+ params): Estimate lattice period, dither range, tile offset |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate lattice period, dither range, tile offset |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate lattice period, dither range, tile offset |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct lattice parameters, stitch registration |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct lattice parameters, stitch registration |
| **M2** Compound | Compound correction with rho measurement: Correct lattice parameters, stitch registration |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct lattice parameters, stitch registration |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct lattice parameters, stitch registration |

### Correction Targets
- **Expected rho**: TBD

### Improvement Roadmap
Compare Bessel vs lattice modes; add multi-view registration.

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
