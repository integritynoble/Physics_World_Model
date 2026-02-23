# Confocal Live-Cell Microscopy (`confocal_livecell`)

**Category**: Microscopy | **Canonical DAG**: C --> D | **Carrier**: Photon
**Current Maturity**: M1 | **Forward Model**: linear_operator | **Default Solver**: richardson_lucy

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Pinhole size, scan speed vs photobleaching |
| **M1** Synthetic | Prompt tested with synthetic data validation: Pinhole size, scan speed vs photobleaching |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Pinhole size, scan speed vs photobleaching |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Pinhole size, scan speed vs photobleaching |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Confocal Live-Cell Microscopy |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Deconvolution + motion correction |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Deconvolution + motion correction |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Deconvolution + motion correction |
| **M3** Real Data | Real experimental data with measured mismatch: Deconvolution + motion correction |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Deconvolution + motion correction |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| PSF sigma | 1.5 | [0.8, 3.0] | px |
| Drift rate | 0.1 | [0, 1.0] | px/frame |
| Bleaching rate | 0.01 | [0, 0.1] | per frame |
| Pinhole misalignment | 0 | [0, 0.5] | AU offset |

### Solvers & Expected Performance
- **Solver**: richardson_lucy

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for C --> D: Estimate drift rate, PSF, bleaching curve |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate drift rate, PSF, bleaching curve |
| **M2** Compound | Compound parameter identification (3+ params): Estimate drift rate, PSF, bleaching curve |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate drift rate, PSF, bleaching curve |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate drift rate, PSF, bleaching curve |

### True-Spec Parameters
PSF, drift trajectory, bleaching curve, pinhole offset

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct drift, update PSF for live conditions |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct drift, update PSF for live conditions |
| **M2** Compound | Compound correction with rho measurement: Correct drift, update PSF for live conditions |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct drift, update PSF for live conditions |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct drift, update PSF for live conditions |

### Correction Targets
- **Expected rho**: >= 0.75

### Improvement Roadmap
Add sample-induced aberration, compound drift+bleaching.

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
