# Ptychographic Imaging (`ptychography`)

**Category**: Coherent Imaging | **Canonical DAG**: M --> P --> D | **Carrier**: Electron/Photon
**Current Maturity**: M3 | **Forward Model**: nonlinear_operator | **Default Solver**: epie

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Probe size, overlap ratio, scan pattern, coherence |
| **M1** Synthetic | Prompt tested with synthetic data validation: Probe size, overlap ratio, scan pattern, coherence |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Probe size, overlap ratio, scan pattern, coherence |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Probe size, overlap ratio, scan pattern, coherence |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Ptychographic Imaging |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: ePIE under probe position error, defocus, aberration |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: ePIE under probe position error, defocus, aberration |
| **M2** Compound | Compound mismatch (3+ params simultaneously): ePIE under probe position error, defocus, aberration |
| **M3** Real Data | Real experimental data with measured mismatch: ePIE under probe position error, defocus, aberration |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: ePIE under probe position error, defocus, aberration |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Probe position error | 0 | [-5, 5] each | px |
| Defocus | 0 | [-50, 50] | nm |
| Partial coherence | 1.0 | [0.7, 1.0] | - |

### Solvers & Expected Performance
- **Solver**: epie
- **Validated baseline**: ePIE +7.09 dB, rho = 1.00

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for M --> P --> D: Estimate probe positions, aberration coefficients |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate probe positions, aberration coefficients |
| **M2** Compound | Compound parameter identification (3+ params): Estimate probe positions, aberration coefficients |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate probe positions, aberration coefficients |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate probe positions, aberration coefficients |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct positions; rho=100%, +7.09 dB |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct positions; rho=100%, +7.09 dB |
| **M2** Compound | Compound correction with rho measurement: Correct positions; rho=100%, +7.09 dB |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct positions; rho=100%, +7.09 dB |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct positions; rho=100%, +7.09 dB |

### Correction Targets
- **Expected rho**: 1.00

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
