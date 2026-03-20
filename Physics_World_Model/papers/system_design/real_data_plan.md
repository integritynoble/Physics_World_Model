# Real-Data Validation Expansion Plan

## Goal
Add 5 more real-data modalities to upgrade validation pyramid from 6 → 11 real-data modalities.
This directly addresses Nature reviewer concern #2 (thin real-data validation).

---

## Recommended Datasets (ranked by feasibility & impact)

### 1. Lensless Imaging — DiffuserCam (Waller Lab)
**Why:** Directly validates our novel lensless system design ($C \to D$) on real hardware.
The highest-impact addition because it bridges simulation and real hardware.

| Field | Details |
|-------|---------|
| **Dataset** | DiffuserCam Lensless MIRFlickr |
| **Size** | 25,000 paired images (measurement + ground truth) |
| **Includes** | Calibrated PSF at multiple exposures |
| **Resolution** | 270×480 raw measurements |
| **Download** | https://huggingface.co/datasets/bezzam/DiffuserCam-Lensless-Mirflickr-Dataset |
| **Alt. source** | https://waller-lab.github.io/LenslessLearning/dataset.html |
| **Toolkit** | https://github.com/LCAV/LenslessPiCam (ADMM, FISTA implementations) |
| **Our chain** | $C \to D$ (lensless) |
| **Expected PSNR** | ~25-30 dB (real data harder than simulation's 43.7 dB) |
| **Effort** | Low — PSF provided, just need to run our reconstructor |

### 2. Ultrasound — PICMUS (IEEE IUS 2016)
**Why:** Upgrades current "self-reference" ultrasound to proper benchmark with raw RF data.

| Field | Details |
|-------|---------|
| **Dataset** | Plane-wave Imaging Challenge in Medical UltraSound |
| **Size** | 4 phantom + 2 in-vivo datasets |
| **Includes** | Raw RF channel data, imaging parameters |
| **Download** | https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/download |
| **Our chain** | $P \to R \to P \to D$ (ultrasound) |
| **Expected PSNR** | ~25-35 dB (depends on number of plane waves) |
| **Effort** | Medium — need to implement beamforming from RF data |

### 3. Light Field — Stanford Lytro Archive
**Why:** Validates light field modality ($M \to C \to S \to D$) with real lenslet data.

| Field | Details |
|-------|---------|
| **Dataset** | Stanford Lytro Light Field Archive |
| **Size** | 118 light field images from Lytro Illum |
| **Includes** | RAW (decoded ESLF) files + processed images |
| **Download** | http://lightfields.stanford.edu/LF2016.html |
| **Toolkit** | MATLAB Light Field Toolbox for decoding |
| **Our chain** | $M \to C \to S \to D$ (light field) |
| **Expected PSNR** | ~28-32 dB |
| **Effort** | Medium — need lenslet calibration + sub-aperture extraction |

### 4. OCT — Duke SD-OCT + MOZART
**Why:** Upgrades OCT from synthetic to real raw interferogram data.

| Field | Details |
|-------|---------|
| **Dataset** | Duke SD-OCT (Farsiu lab) |
| **Size** | 45 patients, volumetric scans |
| **Includes** | Raw + denoised pairs |
| **Download** | https://people.duke.edu/~sf59/software.html |
| **Alt. toolkit** | Vis-OCT Explorer (raw interferogram → volume) |
| **Code** | https://github.com/orlyliba/OCT_Reconstruction_and_Spectral_Analysis |
| **Our chain** | $P + P \to \Sigma \to D$ (OCT) |
| **Expected PSNR** | ~28-35 dB |
| **Effort** | Medium-High — need spectral-domain processing pipeline |

### 5. Photoacoustic Imaging — IEEE DataPort
**Why:** Adds a new hybrid carrier family (optical excitation + acoustic detection).

| Field | Details |
|-------|---------|
| **Dataset** | Photoacoustic Source Detection and Reflection Artifact Dataset |
| **Size** | Experimental phantom data + trained models |
| **Download** | https://ieee-dataport.org/open-access/photoacoustic-source-detection-and-reflection-artifact-deep-learning-dataset |
| **10-algorithm benchmark** | Prakash et al., J. Biophotonics 2024 |
| **Our chain** | $\Lambda \to P \to D$ (photoacoustic) |
| **Expected PSNR** | ~20-28 dB |
| **Effort** | Medium — need photoacoustic forward model |

---

## Alternative / Backup Datasets

### 5b. Electron Ptychography — Zenodo 4D-STEM
Already referenced in paper as `zenodo4dstem`. Raw diffraction patterns from SrTiO3.
- Download: https://doi.org/10.5281/zenodo.5113449
- Our chain: $M \to P \to D$
- Need: ptychographic phase retrieval implementation

### 5c. SIM — SIMnoise / OpenSIM
- Download: https://data.4tu.nl/articles/_/12942932
- Limited to simulated raw SIM stacks (less impactful for Nature)

---

## Implementation Priority

**Phase 1 (1-2 days):** DiffuserCam lensless — lowest effort, highest impact
**Phase 2 (2-3 days):** PICMUS ultrasound — proper RF beamforming
**Phase 3 (3-5 days):** Stanford light field — lenslet calibration needed
**Phase 4 (3-5 days):** Duke OCT — interferogram processing
**Phase 5 (optional):** Photoacoustic or ptychography

---

## Impact on Paper

Current: 6 real-data modalities (CT, MRI, CASSI, CACTI, ultrasound†, e-ptychography†)
  † = self-reference metric only

After Phase 1-4: 10 real-data modalities (adding lensless, ultrasound-RF, light field, OCT)
  - Upgrades ultrasound from self-reference to proper benchmark
  - Adds 3 entirely new real-data validations
  - Validates 1 novel system design (lensless) on real hardware

Validation pyramid becomes: 173 → 39 → 10 (was 6)
