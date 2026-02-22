# Contribution Package: Xin Yuan

**Affiliation:** School of Engineering, Westlake University, Hangzhou, China
**Role:** Co-author (already confirmed)
**Expertise:** GAP-TV, EfficientSCI, compressive sensing reconstruction algorithms, CASSI/CACTI forward models
**Paper:** "Eleven Primitives and Three Gates: The Universal Structure of Computational Imaging" --- Nature submission

---

## Overview

As a co-author, Prof. Yuan's contributions center on validating the reconstruction algorithm protocols, confirming the CASSI/CACTI forward model specifications, and reviewing the mismatch parameter characterizations that underpin the 5-parameter mismatch model used throughout the paper.

---

## Specific Tasks

### Task 1: Validate Reconstruction Algorithm Parameters

Review and confirm the experimental parameters used for all CASSI- and CACTI-related solvers:

| Solver | Modality | Key parameters to validate |
|--------|----------|---------------------------|
| GAP-TV | CASSI | 200 iterations, TV regularization weight, convergence criterion |
| GAP-TV | CACTI | 50 iterations, TV regularization weight |
| EfficientSCI | CACTI | Pre-trained checkpoint (strict=False for CR=10 on CR=8 weights), input normalization |
| PnP-FFDNet | CACTI | 50 iterations, FFDNet denoiser noise level, step size |
| ELP-Unfolding | CACTI | Pre-trained checkpoint, unrolling depth |

**Deliverable:** Written confirmation that the solver parameters match published defaults or justified deviations, with brief rationale for any discrepancies.

### Task 2: Validate Forward Model Specifications

Confirm the CASSI and CACTI forward model specifications used in the OperatorGraph templates:

**CASSI (5-parameter mismatch model):**
- Mask spatial shift: dx = 0.5 px, dy = 0.3 px
- Mask rotation: theta = 0.1 degrees
- Dispersion slope: a_1 = 2.02 pixels/band
- Dispersion axis angle: alpha = 0.15 degrees
- Confirm that these values represent realistic assembly errors for DD-CASSI instruments.

**CACTI:**
- Temporal mask timing offset as the primary mismatch parameter
- Compression ratio CR = 8 (simulation) and CR = 10 (real data)
- Confirm that EfficientSCI with strict=False loading is an acceptable experimental choice for CR=10 evaluation.

**Deliverable:** Written confirmation of forward model correctness or suggested corrections.

### Task 3: Review Real-Data Experimental Protocol

Validate the real-data experimental setup:

**CASSI real data (TSA dataset):**
- 5 scenes, 660 x 660 spatial, 28 spectral bands, step = 2
- Software mask perturbation: dx = 0.5 px, dy = 0.3 px
- Solvers: GAP-TV (200 iter), HDNet (pre-trained, full spatial), MST-S/L (pre-trained, center-cropped to 256 x 256)
- Key finding: GAP-TV residual ratio 1.8x; MST-S/L show ~1.0x on real data (pre-existing mask imperfections absorb perturbation)

**CACTI real data (EfficientSCI dataset):**
- 4 scenes (duomino, hand, pendulumBall, waterBalloon), 512 x 512, CR = 10
- Mask stored separately from measurement data
- Software mask perturbation: dx = 0.5 px, dy = 0.3 px
- Key finding: GAP-TV residual ratio 10.4x; PnP-FFDNet 2.0x

**Deliverable:** Written confirmation that the protocol and interpretation are scientifically sound.

### Task 4: Manuscript Review

Review the following manuscript sections for technical accuracy:

1. Main text: "Empirical Validation" section (CASSI and CACTI subsections)
2. Main text: "Hardware validation" section (CASSI and CACTI real data)
3. Online Methods: CASSI configuration, CACTI configuration, CASSI real-data configuration, CACTI real-data configuration
4. Supplementary: Tables S1--S4 (per-scene PSNR/SSIM/SAM results)
5. Supplementary: Note 15 (cross-residual analysis for CASSI and CACTI)

**Deliverable:** Tracked-changes or annotated comments on the relevant sections.

---

## Estimated Effort

| Task | Estimated time |
|------|---------------|
| Task 1: Solver parameter validation | 0.5 day |
| Task 2: Forward model validation | 0.5 day |
| Task 3: Real-data protocol review | 0.5 day |
| Task 4: Manuscript review | 1--1.5 days |
| **Total** | **2--3 days** |

---

## Author Contributions Statement (Draft)

For the "Author Contributions" section of the manuscript:

> X.Y. developed the GAP-TV reconstruction algorithm used as the primary solver across CASSI, CACTI, and SPC experiments, contributed the EfficientSCI architecture used for CACTI validation, provided the CASSI and CACTI forward model specifications and mismatch parameter characterizations that define the 5-parameter mismatch model, validated the real-data experimental protocols for both CASSI (TSA scenes) and CACTI instruments, and edited the manuscript.

Please review and suggest modifications to this statement.

---

## ICMJE Authorship Criteria Mapping

| ICMJE criterion | Satisfied by |
|----------------|-------------|
| 1. Substantial contribution to conception/design and interpretation of data | Development of GAP-TV and EfficientSCI algorithms used throughout; CASSI/CACTI forward model specifications; mismatch parameter characterization |
| 2. Critical revision for intellectual content | Review of CASSI/CACTI experimental sections, solver parameter validation, real-data protocol validation |
| 3. Final approval | Review and approval of final manuscript |
| 4. Accountability | Agreement to ensure accuracy of reconstruction algorithm descriptions and CASSI/CACTI experimental results |

---

## Timeline

| Milestone | Target |
|-----------|--------|
| Manuscript draft shared | Immediately upon request |
| Tasks 1--3 completed | 1 week after draft review |
| Task 4 (manuscript review) completed | 1 week after Tasks 1--3 |
| Final approval | 1 week after revision |

---

## Contact

Chengshuai Yang
NextGen PlatformAI C Corp
integrityyang@gmail.com

Code and manuscript: https://github.com/integritynoble/Physics_World_Model
