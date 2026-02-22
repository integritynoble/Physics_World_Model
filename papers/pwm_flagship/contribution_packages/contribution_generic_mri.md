# Contribution Package: MRI Physicist (Generic)

**Target profile:** MRI physicist or engineer with access to a research MRI scanner (1.5T or 3T) and the ability to acquire multi-coil brain scans under a research protocol
**Paper:** "Eleven Primitives and Three Gates: The Universal Structure of Computational Imaging" --- Nature submission

---

## Overview

We invite an MRI physicist to provide controlled multi-coil brain scan data demonstrating that coil sensitivity mismatch becomes the dominant reconstruction failure mode under clinically relevant parallel imaging acceleration. This contribution would extend the paper's hardware validation to the nuclear spin carrier family with physical (not simulated) coil repositioning, completing the cross-carrier validation across all four carrier families validated in the paper (optical photons, X-ray photons, electrons, nuclear spins).

---

## Specific Task: Multi-Coil Brain Scan with Controlled Coil Repositioning

### Objective

Demonstrate that physical coil repositioning between scans produces reconstruction degradation consistent with the Triad Decomposition's Gate 3 predictions, and that the degradation severity scales with acceleration factor as predicted.

### Protocol

**Step 1: Baseline acquisition**
- Acquire a fully sampled multi-coil brain dataset using a standard head coil (8+ channels preferred; 32-channel ideal).
- Pulse sequence: T1-weighted MPRAGE or T2-weighted TSE, standard clinical parameters.
- Acquire coil sensitivity maps via a low-resolution calibration scan (standard vendor prescan or dedicated calibration sequence).
- Record: raw multi-coil k-space data (if accessible), reconstructed DICOM images, coil sensitivity maps, scan parameters.

**Step 2: Coil repositioning**
- Without moving the subject (or using a stable phantom), physically reposition the head coil by a small controlled amount:
  - Condition A: Slight lateral shift (~5 mm)
  - Condition B: Slight superior/inferior shift (~5 mm)
  - Condition C: Slight rotation (~2--3 degrees)
- Re-acquire a fully sampled dataset at each repositioned condition.
- Re-acquire coil sensitivity maps at each repositioned condition.
- The key comparison is: reconstruct repositioned data using the *original* (pre-repositioning) sensitivity maps vs. the *updated* (post-repositioning) maps.

**Step 3: Retrospective undersampling**
- From the fully sampled data, generate retrospectively undersampled datasets at:
  - R = 2 (50% k-space lines retained, Cartesian uniform undersampling)
  - R = 4 (25% k-space lines retained)
  - R = 6 (17% k-space lines retained, if SNR permits)
- Reconstruct each undersampled dataset with both the original and updated sensitivity maps.

**Step 4: Data delivery**
- Provide raw multi-coil k-space data (or DICOM images + sensitivity maps) for each condition.
- We handle all SENSE/GRAPPA reconstruction, PSNR/SSIM computation, and PWM pipeline analysis.

---

## What Is Needed

| Item | Specification |
|------|--------------|
| Scanner access | 2--3 days (1.5T or 3T research MRI scanner) |
| Subject/phantom | Healthy volunteer (with IRB approval) or structural brain phantom |
| Head coil | Multi-channel (8+ coils; 32-channel preferred) |
| Repositioning method | Manual coil repositioning with measured displacement (~5 mm shift or ~2--3 degree rotation) |
| Deliverables | Raw multi-coil k-space (ISMRMRD format preferred, or vendor-specific raw data), coil sensitivity maps, scan parameters |

**No software development is required.** We process all data through the PWM pipeline.

---

## Why R = 4+ Is Critical

The paper's current MRI results show that Gate 3 severity scales with acceleration factor:

| Acceleration | Coil sensitivity error | PSNR degradation | Gate 3 dominant? |
|-------------|----------------------|-------------------|-----------------|
| R = 2 | 5% | 0.5 dB | Marginal (coil redundancy absorbs errors) |
| R = 4 | 5% | 1.75--7.14 dB | Yes |
| R = 4 | 3% | ~1.0 dB | Threshold region |

At R = 2, the encoding matrix is well-conditioned and coil redundancy absorbs sensitivity errors, producing only 0.5 dB degradation. This is consistent with the Gate 3 dominance condition (Theorem 2 in the main text): Gate 3 requires sufficiently large condition number kappa(H). At R = 4 and above, the condition number increases and Gate 3 becomes binding.

**The key prediction to test:** Physical coil repositioning at R = 4 should produce 1.75--7.14 dB degradation (depending on the magnitude of sensitivity change), with degradation increasing monotonically with acceleration factor. At R = 2, degradation should be less than 0.5 dB.

---

## Expected Outcomes

### Predictions from simulation

| Metric | Prediction | Simulation basis |
|--------|------------|-----------------|
| PSNR degradation at R = 4, 5% sensitivity mismatch | 1.75--7.14 dB | Multi-coil simulation (Supplementary Note 11) |
| PSNR degradation at R = 2, 5% sensitivity mismatch | < 0.5 dB | M4Raw real k-space validation |
| Degradation trend with R | Monotonically increasing | Condition number scaling |
| Oracle recovery at R = 4 | rho = 0.20 [0.15, 0.26] | Sensitivity re-estimation |
| Dominant artifact | SENSE/GRAPPA ghost (aliasing along phase-encode direction) | Standard parallel imaging failure mode |

### What confirmation means

- **If confirmed at R >= 4:** The Triad Decomposition extends to the nuclear spin carrier family with hardware evidence, and the acceleration-dependent Gate 3 scaling is validated.
- **If R = 2 shows minimal degradation:** This confirms the predicted boundary condition --- well-conditioned encodings absorb mismatch, consistent with the Gate 3 dominance theorem.
- **Clinical relevance:** Modern clinical MRI routinely uses R = 4--8 acceleration. The finding that coil sensitivity mismatch is the dominant failure mode at these acceleration factors has direct implications for clinical scan planning and coil QA.

---

## ICMJE Authorship Criteria Mapping

| ICMJE criterion | Satisfied by |
|----------------|-------------|
| 1. Substantial contribution to acquisition of data | Execution of controlled multi-coil brain scans with physical coil repositioning; provision of raw k-space data and sensitivity maps |
| 2. Critical revision for intellectual content | Review of MRI experimental descriptions, interpretation of coil sensitivity mismatch results, clinical relevance assessment |
| 3. Final approval | Review and approval of final manuscript |
| 4. Accountability | Agreement to ensure accuracy of MRI hardware validation results |

---

## Timeline

| Milestone | Target |
|-----------|--------|
| Agreement to collaborate | Within 1 week of invitation |
| IRB approval (if volunteer scan) | May already be covered under existing protocol |
| MRI scans completed | 2--3 weeks after agreement |
| Raw data delivered | Within 1 week of scan completion |
| PWM analysis completed and shared | 3--5 days after data receipt |
| Manuscript section drafted | 1 week after analysis |
| Review and approval | 1 week after draft shared |

**Total estimated effort: 2--3 days of scanner time + 1--2 hours of manuscript review.**

---

## Contact

Chengshuai Yang
NextGen PlatformAI C Corp
integrityyang@gmail.com

Code and manuscript: https://github.com/integritynoble/Physics_World_Model
