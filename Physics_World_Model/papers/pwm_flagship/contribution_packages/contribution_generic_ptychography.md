# Contribution Package: Electron Microscopist (Generic)

**Target profile:** Electron microscopist with access to a 4D-STEM-capable transmission electron microscope and experience with ptychographic reconstruction
**Paper:** "Eleven Primitives and Three Gates: The Universal Structure of Computational Imaging" --- Nature submission

---

## Overview

We invite an electron microscopist to provide controlled 4D-STEM scan data demonstrating that probe position jitter (stage drift) is the dominant reconstruction failure mode in electron ptychography, consistent with the Triad Decomposition's Gate 3 prediction. The paper already includes validation on public 4D-STEM data from SrTiO3 (Zenodo 5113449) showing 16.1 dB phase degradation under position jitter, with >99.9% oracle recovery. Hardware validation with intentionally introduced stage drift would provide definitive confirmation on a controlled instrument.

---

## Specific Task: 4D-STEM Scan with Controlled Stage Drift

### Objective

Confirm that known probe position errors in 4D-STEM data produce the phase reconstruction degradation predicted by PWM simulation, and that position correction achieves near-complete recovery.

### Protocol

**Step 1: Baseline acquisition**
- Acquire a 4D-STEM dataset on a crystalline specimen (e.g., SrTiO3 [001], Si [110], or any well-characterized thin crystal).
- Standard 4D-STEM parameters: 128 x 128 or 256 x 256 scan grid, convergence semi-angle appropriate for the specimen, direct electron detector (preferred) or scintillator camera.
- Record: 4D-STEM datacube (scan_x, scan_y, kx, ky), probe convergence angle, accelerating voltage, specimen thickness estimate, nominal scan step size.

**Step 2: Controlled position perturbation**
- Introduce known probe position errors by one or more of the following methods:
  - (a) **Software perturbation (simplest):** Reconstruct the baseline dataset using intentionally shifted scan position coordinates (0.5, 1.0, 2.0, 4.0 pixel jitter, applied as random Gaussian offsets to nominal positions).
  - (b) **Scan coil offset:** If accessible, apply a known DC offset to the scan coils between acquisitions to shift the entire scan grid by a controlled amount.
  - (c) **Intentional drift acquisition:** Acquire a second dataset with the stage drift compensation disabled (or with a known delay introduced), so that natural stage drift accumulates during the scan.

**Step 3: Phase reconstruction**
- Reconstruct each dataset using a standard ptychographic algorithm (ePIE, WDD, or SSB).
- For software-perturbation experiments: reconstruct with both the nominal (unperturbed) and perturbed scan positions.
- Compute the integrated center-of-mass (iCoM) phase as a fast proxy for reconstruction quality.

**Step 4: Data delivery**
- Provide: 4D-STEM datacube(s), scan position files, reconstruction parameters.
- We handle all PWM pipeline analysis: position correction, phase PSNR computation, Gate 3 diagnosis.

---

## What Is Needed

| Item | Specification |
|------|--------------|
| Microscope time | 1--2 days (aberration-corrected TEM with 4D-STEM capability preferred, but not required) |
| Specimen | Any well-characterized crystalline thin film (SrTiO3, Si, GaN, etc.) |
| Detector | Direct electron detector preferred (e.g., EMPAD, Medipix, DEF); scintillator camera acceptable |
| Position jitter magnitudes | 0.5, 1.0, 2.0, 4.0 pixel-equivalents of scan step |
| Deliverables | 4D-STEM datacube (HDF5 or binary), scan position coordinates, microscope parameters |

**No software development is required.** We process all data through the PWM pipeline.

---

## Expected Outcomes

### Predictions from simulation

Based on validation with real SrTiO3 4D-STEM data (Zenodo 5113449; 300 kV, 128 x 128 scan):

| Metric | Prediction | Simulation basis |
|--------|------------|-----------------|
| iCoM phase PSNR at 0.5 px jitter | ~35.5 dB (minimal degradation) | SrTiO3 validation |
| iCoM phase PSNR at 2.0 px jitter | ~25 dB (moderate degradation) | SrTiO3 validation |
| iCoM phase PSNR at 4.0 px jitter | ~19.4 dB (severe degradation) | SrTiO3 validation |
| Total PSNR loss (0 to 4.0 px) | 16.1 dB | SrTiO3 validation |
| Degradation trend | Monotonic with jitter magnitude | Confirmed in simulation |
| Oracle recovery ratio | >99.9% (rho ~ 1.00) | Position correction is a well-posed 1D problem per scan point |

### The falsifiable prediction

The paper makes a specific falsifiable prediction for electron ptychography (Discussion section):

> "Position errors exceeding 1/10 of the probe diameter should trigger Gate 3 dominance."

This prediction was confirmed on public SrTiO3 data (+5 to +16 dB correction). Hardware validation with controlled stage drift on your instrument would provide independent confirmation (or falsification).

### What confirmation means

- **If confirmed:** The Triad Decomposition is validated on the electron carrier family with hardware evidence, completing cross-carrier validation (optical photons, X-rays, electrons, nuclear spins).
- **If not confirmed:** The discrepancy would reveal specimen-dependent or instrument-dependent factors not captured by the current model, informing framework revision.
- **Practical implication:** Automated position correction integrated into 4D-STEM reconstruction pipelines could recover substantial phase quality without requiring expensive hardware drift correction systems.

---

## ICMJE Authorship Criteria Mapping

| ICMJE criterion | Satisfied by |
|----------------|-------------|
| 1. Substantial contribution to acquisition of data | Execution of controlled 4D-STEM experiments with known position perturbation; provision of 4D-STEM datacubes |
| 2. Critical revision for intellectual content | Review of ptychography experimental descriptions, interpretation of position jitter results, domain expertise on electron microscopy artifacts |
| 3. Final approval | Review and approval of final manuscript |
| 4. Accountability | Agreement to ensure accuracy of 4D-STEM hardware validation results |

---

## Timeline

| Milestone | Target |
|-----------|--------|
| Agreement to collaborate | Within 1 week of invitation |
| 4D-STEM experiments completed | 2--3 weeks after agreement (1--2 days of microscope time) |
| Data delivered | Within 1 week of experiment completion |
| PWM analysis completed and shared | 3--5 days after data receipt |
| Manuscript section drafted | 1 week after analysis |
| Review and approval | 1 week after draft shared |

**Total estimated effort: 1--2 days of microscope time + 1--2 hours of manuscript review.**

---

## Data Format

```
ptychography_hardware_validation/
  baseline/
    datacube.hdf5            # 4D-STEM data (scan_x, scan_y, kx, ky)
    scan_positions.csv       # nominal (x, y) for each scan point
    metadata.json            # voltage, convergence angle, step size, detector info
  jitter_0p5px/
    datacube.hdf5            # same acquisition with position perturbation
    scan_positions_true.csv  # actual (perturbed) positions, if known
    metadata.json
  jitter_1p0px/
    ...
  jitter_2p0px/
    ...
  jitter_4p0px/
    ...
```

For software-perturbation experiments (method a), only the baseline datacube is needed; we apply position offsets in post-processing.

---

## Contact

Chengshuai Yang
NextGen PlatformAI C Corp
integrityyang@gmail.com

Code and manuscript: https://github.com/integritynoble/Physics_World_Model
