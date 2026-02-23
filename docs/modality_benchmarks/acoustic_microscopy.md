# Scanning Acoustic Microscopy (SAM) (`acoustic_microscopy`)

**Category**: Industrial Inspection | **Canonical DAG**: P --> D | **Carrier**: Acoustic
**Current Maturity**: M0 | **Forward Model**: nonlinear_operator | **Default Solver**: c_scan_recon

---

## B1: Design (Prompt + Original-Spec --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | Design template: Frequency (50-2000 MHz), lens geometry, coupling |
| **M1** Synthetic | Prompt tested with synthetic data validation: Frequency (50-2000 MHz), lens geometry, coupling |
| **M2** Compound | Multiple design variants; multi-constraint Pareto optimization for Frequency (50-2000 MHz), lens geometry, coupling |
| **M3** Real Data | Grounded in real experimental/clinical protocols: Frequency (50-2000 MHz), lens geometry, coupling |
| **M4** Adversarial | Adversarial/edge-case prompts with contradictory requirements for Scanning Acoustic Microscopy (SAM) |

---

## B2: Forward + Reconstruct (Spec --> Reconstruction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Forward model template with nominal parameters: C-scan under defocus, coupling variation |
| **M1** Synthetic | Single-parameter mismatch on synthetic data: C-scan under defocus, coupling variation |
| **M2** Compound | Compound mismatch (3+ params simultaneously): C-scan under defocus, coupling variation |
| **M3** Real Data | Real experimental data with measured mismatch: C-scan under defocus, coupling variation |
| **M4** Adversarial | Red Team worst-case mismatch injection optimized to maximize failure: C-scan under defocus, coupling variation |

### Mismatch Parameters
| Parameter | Nominal | Mismatch Range | Unit |
|-----------|---------|----------------|------|
| Coupling medium speed | 1480 | [1450, 1550] | m/s |
| Focus depth error | 0 | [-20, 20] | um |
| Lens aberration | 0 | [0, 0.2] | waves |
| Gate position error | 0 | [-5%, 5%] | - |

### Solvers & Expected Performance
- **Solver**: c_scan_recon

---

## B3: System Identification (Dataset + Prompt --> Spec)

| Level | Specification |
|-------|--------------|
| **M0** Template | DAG template identification for P --> D: Estimate focal position, coupling impedance, velocity |
| **M1** Synthetic | Synthetic True-Spec with single-parameter identification: Estimate focal position, coupling impedance, velocity |
| **M2** Compound | Compound parameter identification (3+ params): Estimate focal position, coupling impedance, velocity |
| **M3** Real Data | Real True-Spec from calibration experiments: Estimate focal position, coupling impedance, velocity |
| **M4** Adversarial | Adversarial identification under unknown configuration: Estimate focal position, coupling impedance, velocity |

---

## B4: Correct + Diagnose (Dataset + Spec --> Correction + Feedback)

| Level | Specification |
|-------|--------------|
| **M0** Template | Correction template: Correct defocus, coupling normalization |
| **M1** Synthetic | Single-parameter correction on synthetic data: Correct defocus, coupling normalization |
| **M2** Compound | Compound correction with rho measurement: Correct defocus, coupling normalization |
| **M3** Real Data | Real data correction (target rho >= 0.80): Correct defocus, coupling normalization |
| **M4** Adversarial | Adversarial correction with live feedback loop under compound failures: Correct defocus, coupling normalization |

### Correction Targets
- **Expected rho**: TBD

---

## References
- [Detailed Benchmarks](../pwm_modality_benchmarks_detailed.md)
- [Medical Physicist Targets](../pwm_medical_physicist_targets.md)
- [Modality Registry](../imaging_modalities.md)
