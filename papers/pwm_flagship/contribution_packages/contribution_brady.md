# Contribution Package: David J. Brady

**Affiliation:** Department of Electrical and Computer Engineering, University of Arizona, Tucson, AZ, USA
**Expertise:** Coded aperture snapshot spectral imaging (CASSI), coded aperture compressive temporal imaging (CACTI), computational camera design
**Paper:** "Eleven Primitives and Three Gates: The Universal Structure of Computational Imaging" --- Nature submission

---

## Overview

We invite Prof. Brady to provide controlled hardware validation on DD-CASSI and/or CACTI instruments. As the inventor of both modalities, his contribution would convert the current software-perturbation results into definitive hardware-in-the-loop evidence, providing the strongest possible validation that the Triad Decomposition's mismatch-dominance finding holds on physical instruments under real manufacturing tolerances and environmental conditions.

---

## Specific Task: Physical Mask Displacement Experiment

### Objective

Confirm that sub-pixel mask displacement on real CASSI and CACTI instruments produces the mismatch sensitivity predicted by the PWM framework, and that autonomous calibration recovers the predicted fraction of the oracle correction ceiling.

### Protocol

The experiment follows the controlled hardware protocol specified in the manuscript (Online Methods, "Controlled hardware experiment protocol"):

**Step 1: Baseline acquisition**
- Acquire a reference dataset with the coded aperture mask at its factory-calibrated position.
- Record: raw detector measurement, mask specification file, illumination conditions, exposure parameters.

**Step 2: Controlled mask displacement**
- Physically translate the coded aperture mask by known displacements using a micrometer translation stage.
- Required displacement magnitudes: 0.25, 0.5, 1.0, and 2.0 pixel-equivalents (verify displacement magnitude with the micrometer reading).
- Re-acquire under identical illumination, exposure, and scene conditions at each displacement.
- For each displacement, acquire 2--3 repeat measurements to assess measurement noise.

**Step 3: Data delivery**
- Provide raw measurements and mask files for each condition (baseline + 4 displacement levels).
- We handle all reconstruction, residual analysis, and PWM correction pipeline execution.

**Step 4 (optional): Multi-unit variation study**
- If 2 or more camera units of the same design are available, acquire the same scene on each unit with factory calibration.
- This quantifies the inter-unit mismatch baseline --- a measurement that has never been reported in the literature and would significantly strengthen the paper.

### Instruments

| Instrument | Priority | Displacement axis | Notes |
|-----------|----------|-------------------|-------|
| DD-CASSI | High | x and y (mask plane) | 660 x 660 spatial, 28 spectral bands, step 2 |
| CACTI | High | x and y (temporal mask) | 512 x 512 spatial, CR = 10 |

Either instrument alone is a substantial contribution; both would be ideal.

---

## What Is Needed

| Item | Specification |
|------|--------------|
| Lab time | 1--2 days per instrument |
| Translation stage | Micrometer-precision (sub-pixel, i.e., sub-10 um for typical pixel pitches) |
| Scene | Any spectrally rich target for CASSI; any dynamic scene for CACTI. Standard test targets preferred for reproducibility. |
| Illumination | Stable broadband source (CASSI); stable white-light source (CACTI). Identical conditions across all displacement levels. |
| Deliverables | Raw measurement arrays (TIFF, MAT, or NPY), mask files, micrometer readings, exposure metadata |

**No software development is required.** We process all data through the PWM pipeline and share results for review.

---

## Data Format

```
brady_hardware_validation/
  cassi/
    baseline/
      measurement.mat (or .npy)    # raw detector image
      mask.mat                      # binary coded aperture mask
      metadata.json                 # exposure time, source, micrometer reading
    shift_0p25px/
      measurement.mat
      mask.mat (same as baseline)
      metadata.json                 # micrometer reading confirming displacement
    shift_0p50px/
      ...
    shift_1p0px/
      ...
    shift_2p0px/
      ...
  cacti/
    baseline/
      measurement.mat
      mask.mat
      metadata.json
    shift_0p25px/
      ...
    (same structure)
```

---

## Expected Outcomes

### Predictions from simulation

The PWM framework generates the following testable predictions for hardware validation:

| Modality | Metric | Prediction | Simulation basis |
|----------|--------|------------|-----------------|
| CASSI (GAP-TV) | Measurement residual ratio at 0.5 px | ~1.8x | Real TSA data, software perturbation |
| CASSI (GAP-TV) | Cross-residual at 2.0 px | ~11.1% | Cross-residual analysis (Supplementary Note 15) |
| CACTI (GAP-TV) | Measurement residual ratio at 0.5 px | ~10.4x | Real EfficientSCI data, software perturbation |
| CACTI (GAP-TV) | Cross-residual at 1.0 px | ~462x self/cross dissociation | Cross-residual analysis (Supplementary Note 15) |
| CASSI | Autonomous calibration recovery | 85% of oracle ceiling | Grid-search calibration on real data |
| CACTI | Autonomous calibration recovery | 100% of oracle ceiling | Grid-search calibration on real data |

### What confirmation means

- **If confirmed:** The Triad Decomposition's Gate 3 dominance finding is validated with physical hardware evidence from the inventor of both modalities, substantially strengthening the paper for Nature review.
- **If not confirmed:** The discrepancy itself is scientifically valuable --- it would reveal aspects of real hardware behavior not captured by the software-perturbation protocol, informing framework revision.

### The simulation-to-hardware gap

The current results already reveal an instructive asymmetry:
- CASSI shows *smaller* real degradation (1.8x) than simulation predicts, because as-built masks contain pre-existing manufacturing imperfections that absorb incremental perturbations.
- CACTI shows *larger* real sensitivity (10.4x) than simulation, because temporal mask patterns replicate errors across all compressed frames.

Physical mask displacement data from your instruments would definitively characterize this gap.

---

## ICMJE Authorship Criteria Mapping

| ICMJE criterion | Satisfied by |
|----------------|-------------|
| 1. Substantial contribution to acquisition of data | Execution of controlled mask displacement experiments on CASSI and/or CACTI instruments; provision of raw measurement data under the hardware validation protocol |
| 2. Critical revision for intellectual content | Review of CASSI/CACTI experimental descriptions, hardware validation interpretation, and simulation-to-hardware gap analysis |
| 3. Final approval | Review and approval of final manuscript |
| 4. Accountability | Agreement to ensure accuracy of hardware validation results |

---

## Timeline

| Milestone | Target |
|-----------|--------|
| Agreement to collaborate | Within 1 week of invitation |
| Hardware experiments completed | 2--3 weeks after agreement |
| Raw data delivered | Same day as experiment completion |
| PWM analysis completed and shared | 3--5 days after data receipt |
| Manuscript section drafted | 1 week after analysis |
| Review and approval | 1 week after draft shared |

**Total estimated effort: 1--2 days of lab time + 1--2 hours of manuscript review.**

---

## Contact

Chengshuai Yang
NextGen PlatformAI C Corp
integrityyang@gmail.com

Code and manuscript: https://github.com/integritynoble/Physics_World_Model
