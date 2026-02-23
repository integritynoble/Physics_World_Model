# Atomic Force Microscopy (AFM) (`afm`)

**Category**: Scanning Probe Microscopy | **Canonical DAG**: S --> D | **Carrier**: Mechanical
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: tip_deconvolution

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | "Design AFM scan for semiconductor feature metrology: tapping mode, 1 um scan, 512 lines." |
| **M1** Synthetic | Prompt tested with synthetic data validation: Cantilever spring constant, scan rate, setpoint |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Cantilever spring constant, scan rate, setpoint |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Cantilever spring constant, scan rate, setpoint |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Atomic Force Microscopy (AFM) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: Topography under piezo creep, thermal drift, tip convolution |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: Topography under piezo creep, thermal drift, tip convolution |
| **M2** Compound | Compound mismatch (3+ params simultaneously): Topography under piezo creep, thermal drift, tip convolution |
| **M3** Real Data | Real experimental data with measured mismatch: Topography under piezo creep, thermal drift, tip convolution |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: Topography under piezo creep, thermal drift, tip convolution |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Tip shape convolution | ideal | +/- 30% radius | - |
| Piezo nonlinearity | 0 | [0, 5%] | - |
| Thermal drift | 0 | [0, 1] | nm/s |
| Scanner hysteresis | 0 | [0, 10%] | - |

### Solvers & Expected Performance
- **Solver**: tip_deconvolution

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for S --> D: Estimate piezo coefficients, drift rate, tip shape |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate piezo coefficients, drift rate, tip shape |
| **M2** Compound | Compound parameter identification (3+ params): Estimate piezo coefficients, drift rate, tip shape |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate piezo coefficients, drift rate, tip shape |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate piezo coefficients, drift rate, tip shape |

### True-Spec Parameters
True tip shape, piezo calibration, drift trajectory, hysteresis curve

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct piezo nonlinearity, drift, deconvolve tip |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct piezo nonlinearity, drift, deconvolve tip |
| **M2** Compound | Compound correction with rho measurement: Correct piezo nonlinearity, drift, deconvolve tip |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct piezo nonlinearity, drift, deconvolve tip |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct piezo nonlinearity, drift, deconvolve tip |

### Correction Targets
- **Expected rho**: >= 0.75

### Improvement Roadmap
Add tip deconvolution benchmark, high-speed AFM.

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
