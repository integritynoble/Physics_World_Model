# Contribution Package: Steve B. Jiang

**Affiliation:** Department of Radiation Oncology, UT Southwestern Medical Center, Dallas, TX, USA
**Expertise:** Medical physics, radiation oncology, clinical CT quality assurance, AI in medical imaging
**Paper:** "Eleven Primitives and Three Gates: The Universal Structure of Computational Imaging" --- Nature submission

---

## Overview

We invite Prof. Jiang to provide controlled CT phantom validation demonstrating that center-of-rotation (CoR) mismatch is the dominant reconstruction failure mode in clinical CT, consistent with the Triad Decomposition's Gate 3 dominance prediction. This contribution bridges the paper's cross-modality framework to clinical radiology, establishing that the same mismatch-dominance pattern observed in optical imaging, electron microscopy, and MRI extends to clinical X-ray CT under realistic quality assurance conditions.

---

## Specific Task: CT Phantom Scan with Controlled Center-of-Rotation Offset

### Objective

Confirm that intentional CoR offsets on a clinical CT scanner produce the PSNR degradation and artifact signatures predicted by PWM simulation, and that oracle correction achieves 100% recovery.

### Protocol

**Step 1: Baseline acquisition**
- Scan the ACR CT accreditation phantom (Gammex 464) at the scanner's factory-calibrated CoR position.
- Standard clinical protocol: 120 kVp, standard reconstruction kernel, 5 mm slice thickness.
- Record: raw sinogram data (if accessible) or reconstructed DICOM images, scan parameters.

**Step 2: Controlled CoR displacement**
- Introduce known CoR offsets by one of the following methods:
  - (a) Software recalibration of the scanner's CoR parameter (preferred, if accessible in service mode);
  - (b) Physical phantom displacement on the table by known amounts (1, 2, 4, 8 mm), with the scanner CoR held at the original calibration.
- Required offset magnitudes: 1, 2, 4, and 8 pixel-equivalents (1 pixel ~ 0.5--1.0 mm depending on reconstruction matrix and FOV).
- Acquire at each offset under identical scan parameters.

**Step 3: ACR metric extraction**
- For each acquisition, compute the 10 standard ACR QC metrics (we provide automated extraction scripts, or manual extraction is acceptable):
  1. CT number accuracy --- water (0 +/- 7 HU)
  2. CT number accuracy --- bone (~955 HU)
  3. CT number accuracy --- air (~-1000 HU)
  4. CT number accuracy --- acrylic (~121 HU)
  5. CT number accuracy --- polyethylene (~-96 HU)
  6. Geometric accuracy (+/- 2 mm)
  7. Slice thickness (+/- 1.5 mm)
  8. Uniformity (< 5 HU center-to-edge)
  9. Noise standard deviation
  10. Spatial resolution (> 5 lp/cm)

**Step 4: Data delivery**
- Provide DICOM images (or raw sinograms if available) for each condition.
- We handle all PWM pipeline analysis: FBP/SART reconstruction from sinograms, CoR correction, PSNR/SSIM computation, and artifact characterization.

---

## What Is Needed

| Item | Specification |
|------|--------------|
| Scanner access | 1 day (clinical or research CT scanner) |
| Phantom | ACR CT accreditation phantom (Gammex 464), available in most radiology physics departments |
| CoR displacement | Either service-mode recalibration or physical phantom table offset |
| Offset magnitudes | 1, 2, 4, 8 pixel-equivalents |
| Deliverables | DICOM images at each offset, scan parameter log, optional raw sinogram data |

**No software development is required.** We process all data through the PWM pipeline.

---

## Expected Outcomes

### Predictions from simulation

The PWM framework generates the following testable predictions, based on validation with two public sinogram datasets (FIPS walnut micro-CT: 1200 projections, 2296 detectors; Helsinki Tomography Challenge 2022: 721 projections, 560 detectors):

| Metric | Prediction | Simulation basis |
|--------|------------|-----------------|
| PSNR loss at 4-pixel CoR offset | 8--9 dB | FIPS: 9.4 dB, HTC: 8.0 dB |
| PSNR loss trend | Monotonic with offset magnitude | Confirmed in simulation |
| Oracle recovery ratio | 100% (rho = 1.00, CI [0.99, 1.00]) | 1D parameter, clean minimum |
| Dominant artifact | Characteristic arc/ring artifacts | CoR offset signature |
| ACR metrics affected | Uniformity, geometric accuracy, CT number accuracy | Predicted by Triad Gate 3 analysis |

### What confirmation means

- **If confirmed:** The Triad Decomposition is validated on a clinical imaging modality with direct relevance to the ~40,000 CT scanners in the US that undergo annual ACR accreditation.
- **Clinical QC relevance:** CoR drift is a known but under-monitored failure mode in clinical CT. Demonstration that PWM autonomously detects and quantifies CoR mismatch supports the CT QC Copilot concept described in the supplementary material.

### Clinical translation pathway

The Triad Decomposition maps directly to clinical CT failure categories:

| Triad gate | Clinical CT equivalent | Clinical action |
|-----------|----------------------|----------------|
| Gate 1 (Recoverability) | Protocol design inadequacy (insufficient projections, FOV) | Protocol optimization |
| Gate 2 (Carrier budget) | Dose budget (noise vs. diagnostic need) | Dose adjustment |
| Gate 3 (Operator mismatch) | Scanner calibration drift (HU drift, CoR offset, gain variation) | Recalibration |

In clinical practice, Gate 3 dominates: most QA failures trace to calibration drift, not protocol or dose issues. This is consistent with the paper's cross-modality finding.

---

## ICMJE Authorship Criteria Mapping

| ICMJE criterion | Satisfied by |
|----------------|-------------|
| 1. Substantial contribution to acquisition of data | Execution of controlled CT phantom experiments with intentional CoR offsets; provision of DICOM/sinogram data |
| 2. Critical revision for intellectual content | Review of CT experimental descriptions, clinical QC interpretation, and Triad-to-clinical mapping |
| 3. Final approval | Review and approval of final manuscript |
| 4. Accountability | Agreement to ensure accuracy of CT validation results and clinical interpretation |

---

## Timeline

| Milestone | Target |
|-----------|--------|
| Agreement to collaborate | Within 1 week of invitation |
| Phantom scan completed | 2 weeks after agreement (1 day of scanner time) |
| DICOM data delivered | Same day as scan completion |
| PWM analysis completed and shared | 3--5 days after data receipt |
| Manuscript section drafted | 1 week after analysis |
| Review and approval | 1 week after draft shared |

**Total estimated effort: 1 day of scanner time + 1--2 hours of manuscript review.**

---

## Additional Opportunity: CT QC Copilot Validation

The supplementary material describes a CT QC Copilot that automates ACR phantom analysis with 9 metrics, achieving agreement within 1.2 HU for CT number accuracy and 0.15 mm for geometric accuracy compared to console-reported values. If of interest, we would welcome validation of the Copilot on your department's clinical scanner fleet, which would:

- Provide independent validation of the automated QC pipeline on production clinical scanners.
- Quantify the simulation-to-clinical gap for CT mismatch sensitivity.
- Demonstrate early drift detection (3--6 months before ACR threshold exceedance in simulation).

This is an optional extension that could strengthen a clinical translation narrative.

---

## Contact

Chengshuai Yang
NextGen PlatformAI C Corp
integrityyang@gmail.com

Code and manuscript: https://github.com/integritynoble/Physics_World_Model
