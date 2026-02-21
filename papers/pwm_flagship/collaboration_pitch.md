# Collaboration Pitch: Nature Paper on Universal Computational Imaging

## Paper Title
**"Ten Primitives and Three Gates: The Universal Structure of Computational Imaging"**

*Target: Nature (Article)*

---

## One-Sentence Summary

We prove that every computational imaging forward model decomposes into exactly 10 primitive operators, that operator mismatch universally dominates reconstruction failures, and that a single modality-agnostic calibration pipeline recovers +0.8 to +10.7 dB across 7 modalities and 3 carrier families.

---

## What We Have (Ready Now)

- **Complete theoretical framework**: Finite Primitive Basis Theorem (10 primitives, proven minimal) + Triad Decomposition (3 failure gates)
- **Open-source codebase**: 200+ Python modules, 26 modality templates, 139 completed experiment bundles, full reproducibility infrastructure
- **Simulation validation**: 7 modalities (CASSI, CACTI, SPC, CT, MRI, ptychography, lensless) with 4-scenario evaluation protocol
- **Real-data validation**: 5 modalities across 4 carrier families: CASSI (5 TSA scenes, cross-residual analysis), CACTI (4 EfficientSCI scenes, self- vs. cross-residual dissociation), CT (2 public sinogram datasets: FIPS walnut + HTC 2022, CoR mismatch), electron ptychography (4D-STEM SrTiO₃, 16.1 dB position-jitter degradation), MRI (M4Raw multi-coil brain, SENSE R=2 sensitivity mismatch)
- **Falsifiable predictions**: 2 remaining predictions for SIM and OCT; electron ptychography prediction confirmed (+5 to +16 dB)
- **Companion papers**: InverseNet (ECCV 2026, detailed mismatch analysis), Finite Primitive Theorem (SIAM, formal semantics)

---

## What We Need (Your Contribution)

### Option A: Hardware Experimentalist (Highest Priority)

**What you do:** Provide 2-3 controlled acquisitions on your instrument where you:
1. Acquire a baseline dataset with nominal calibration
2. Physically displace a calibration parameter by a known amount (e.g., mask shift, CoR offset, coil repositioning)
3. Re-acquire under identical conditions

**Instruments of interest:**
- DD-CASSI or SD-CASSI (coded aperture mask shift: 0.25, 0.5, 1.0 px)
- CACTI (temporal mask timing offset)
- Micro-CT (center-of-rotation offset: 1, 3, 5 pixels)
- Clinical MRI (coil repositioning between scans)
- Ptychography beamline (known probe position perturbation)

**What you get:**
- Co-authorship on a Nature paper
- Your instrument provides the controlled hardware validation that elevates simulation results to physical evidence
- The PWM pipeline processes your data without modification --- we handle all analysis
- Your data becomes part of a landmark cross-modality dataset

**Estimated effort:** 1-2 days of instrument time, no software development needed

### Option B: Medical Imaging PI

**What you do:** Provide clinical phantom data (CT ACR phantom or MRI calibration phantom) with controlled mismatch parameters.

**What you get:**
- Co-authorship
- The Triad framework applied to your clinical QC workflow
- A published demonstration that Gate 3 (calibration drift) dominates clinical imaging failures

### Option C: Inverse Problems Theorist

**What you do:** Help strengthen the Finite Primitive Basis Theorem:
- Prove a tighter error bound (current bound is 10-100x conservative)
- Establish algebraic closure properties of the 10 primitives under composition
- Connect the primitive basis to category-theoretic structure

**What you get:**
- Co-authorship on both the Nature paper and the companion SIAM paper
- A new mathematical framework for reasoning about imaging system design

### Option D: Microscopy / Beamline Scientist

**What you do:** Provide real ptychography, SIM, or OCT data with known calibration parameters. Test one of our falsifiable predictions.

**What you get:**
- Co-authorship
- Independent validation (or falsification) of the Gate 3 universality claim on your modality
- The first automated calibration pipeline for your instrument

---

## Key Figures

1. **Periodic Table of Imaging Primitives**: 10 primitives organized by physics-stage family, covering 26+ modalities across 5 carrier families
2. **4-Scenario Bar Chart**: +0.8 to +10.7 dB correction gains across 7 modalities, carrier-agnostic
3. **Basis-Growth Saturation Curve**: K=10 primitives saturates at N=31 modalities --- no new primitive needed for the most recent 19 modalities

---

## Timeline

- **Current status**: Manuscript draft complete, simulation experiments complete, real-data validation on CASSI/CACTI complete
- **What's needed**: 1-2 controlled hardware experiments (4-8 weeks)
- **Target submission**: Nature, Q2 2026

---

## Contact

Chengshuai Yang
NextGen PlatformAI C Corp
integrityyang@gmail.com

Code: https://github.com/integritynoble/Physics_World_Model
