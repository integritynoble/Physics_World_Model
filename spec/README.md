# PWM Spec.md Database

**Physics World Model** — Structured specification database for computational imaging.

## What Is This?

The `spec/` directory is a structured database of imaging specifications covering **4 use cases** across **169 modalities**. Each spec defines the imaging system, forward model, reconstruction algorithms, and runnable code.

## Quick Start

### No API Key (Keyword Matching)

```bash
# Find and run specs by keyword
python spec/keyword_match.py "CT reconstruction"
python spec/keyword_match.py "MRI with mismatch correction"
python spec/keyword_match.py "design lensless imaging system"
python spec/keyword_match.py "simulate CT physics"

# List all available modalities
python spec/keyword_match.py --list

# Run reconstruction directly
python spec/keyword_match.py "CT" --run
```

> **Note (No API Key)**: PWM is using **keyword matching** to find your spec.
> For LLM-guided spec refinement, provide a Gemini 2.5 Flash API key via `--api-key` or
> a key from https://comparegpt.io/modelGateway via `--gateway-key`.

### With LLM API Key (Multi-round Finetuning)
```bash
python spec/keyword_match.py "sparse-view CT with scatter correction" --api-key YOUR_GEMINI_KEY
```

---

## Setup (Local or Colab)

### Local Installation
```bash
git clone https://github.com/integritynoble/Physics_World_Model
cd Physics_World_Model/pwm/public
pip install -e packages/pwm_core/
```

### Google Colab
```python
!git clone https://github.com/integritynoble/Physics_World_Model
import sys
sys.path.insert(0, '/content/Physics_World_Model/pwm/public')
sys.path.insert(0, '/content/Physics_World_Model/pwm/public/packages/pwm_core')
```

### GPU Note
Algorithms marked **GPU** require CUDA. If your machine has no GPU, those algorithms will raise an error — this does **not** affect CPU algorithms, which continue to work normally. Always check the "GPU" column in the algorithm table before selecting a solver.

---

## 4 Use Cases

| # | Use Case | Spec Directory | Description |
|---|----------|---------------|-------------|
| 1 | **Reconstruct** | `spec/01_reconstruct/` | Run reconstruction algorithms on measurement data |
| 2 | **Mismatch + Reconstruct** | `spec/02_mismatch_reconstruct/` | Correct operator mismatch, then reconstruct |
| 3 | **Imaging System Design** | `spec/03_system_design/` | Simulate forward model (with mismatch) + reconstruct |
| 4 | **Scientific Simulation** | `spec/04_simulation/` | Physics simulation examples from PWM papers |

---

## Use Case 1: Reconstruct with Specific Algorithms

Each spec in `spec/01_reconstruct/` contains:
- **System Overview** — physical parameters that influence reconstruction choice
- **Algorithm Catalog** — table of solvers with reference PSNR/SSIM
- **Measurement Data** — use PWM's benchmark data or upload your own
- **Run Button** — copy-paste Python code block to execute

**Key Modalities** (full specs available):
| Spec File | Modality | Best CPU PSNR | Best GPU PSNR |
|-----------|----------|---------------|---------------|
| `ct.md` | X-ray CT | 39.5 dB (PnP-NLM) | 43.5 dB (InDuDoNet) |
| `mri.md` | MRI | 34.2 dB (ESPIRiT) | 37.8 dB (ReconFormer) |
| `cbct.md` | Cone-Beam CT | 34 dB (TV-ADMM) | 38+ dB (DL) |
| `pet.md` | PET | 30 dB (OSEM) | 35+ dB (DL) |
| `spect.md` | SPECT | 28 dB (OSEM) | — |
| `ultrasound.md` | Ultrasound | 32 dB (DAS) | — |
| `oct.md` | OCT | 35 dB (TV) | 40+ dB (DL) |
| `cassi.md` | Coded-Aperture Spectral | 38 dB (GAP-TV) | 42 dB (MST-L) |
| `cacti.md` | Compressed Video | 35 dB (GAP-TV) | 40 dB (EfficientSCI) |
| `lensless.md` | Lensless Imaging | 32 dB (ADMM-TV) | 38 dB (PhysenNet) |
| `phase_retrieval.md` | Phase Retrieval | 36 dB (HIO+ER) | 40 dB (DL) |
| `sim.md` | Structured Illumination | 37 dB (wienerSIM) | 42 dB (DFCAN) |

See `spec/01_reconstruct/INDEX.md` for the complete table of all 169 modalities.

---

## Use Case 2: Mismatch Correction + Reconstruct

Each spec in `spec/02_mismatch_reconstruct/` contains:
- **Modality + mismatch type** (PSF error, center-of-rotation shift, coil mismatch, etc.)
- **Mismatch parameters** with physically plausible ranges
- **Correction method** (grid search, cross-correlation, neural calibration)
- **Reconstruction** after correction

**Key Specs:**
| Spec File | Modality | Mismatch Type | Correction Method |
|-----------|----------|---------------|-------------------|
| `ct_mismatch.md` | CT | Center-of-rotation offset | Cross-correlation |
| `mri_mismatch.md` | MRI | Coil sensitivity error | ESPIRiT |
| `cassi_mismatch.md` | CASSI | Dispersion step error | Grid search |
| `lensless_mismatch.md` | Lensless | PSF shift | Gradient calibration |
| `microscopy_mismatch.md` | Widefield/Confocal | PSF sigma, defocus | Grid search |

See `spec/02_mismatch_reconstruct/INDEX.md` for the full list.

---

## Use Case 3: Imaging System Design

Each spec in `spec/03_system_design/` follows the three-agent pipeline from the PWM System Design paper:
- **DAG** of physical elements (source → interaction → geometry → detector → ADC)
- **Noise model** for each element
- **Mismatch sources** with severity and correction methods
- **Reconstruction algorithm** matched to the forward model

**Key Specs:**
| Spec File | System | Elements | Mismatch Sources |
|-----------|--------|----------|-----------------|
| `ct_system.md` | Sparse-view CT | Tube → Phantom → Geometry → Detector → ADC | Beam hardening, scatter, CoR offset |
| `mri_system.md` | Undersampled MRI | RF coil → k-space → Coil array → ADC | B0 drift, coil mismatch |
| `lensless_system.md` | Lensless Camera | LED → Diffuser → Sensor | PSF shift, background |
| `sim_system.md` | SIM Microscopy | Laser → SLM → Sample → Objective → Camera | Pattern mismatch, defocus |

---

## Use Case 4: Scientific Simulation

Each spec in `spec/04_simulation/` provides:
- Physics equations (forward model)
- Simulation code (NumPy/SciPy)
- Validation against analytical ground truth
- Example from PWM papers

**Key Specs:**
| Spec File | Domain | Simulation Type |
|-----------|--------|----------------|
| `ct_simulation.md` | X-ray CT | Radon transform + Poisson noise |
| `optics_simulation.md` | Diffraction Optics | Fresnel propagation |
| `mri_simulation.md` | MRI | Bloch equation + k-space sampling |
| `wave_simulation.md` | Acoustics/Seismic | Wave equation (FDTD) |

---

## File Structure

```
spec/
├── README.md                        # This file
├── keyword_match.py                 # Keyword matching CLI (no API key path)
├── 01_reconstruct/
│   ├── INDEX.md                     # Complete modality table (169 entries)
│   ├── ct.md                        # CT full spec
│   ├── mri.md                       # MRI full spec
│   ├── cbct.md, pet.md, spect.md, ultrasound.md, oct.md
│   ├── cassi.md, cacti.md, lensless.md, phase_retrieval.md, sim.md
│   └── _template.md                 # Template for unlisted modalities
├── 02_mismatch_reconstruct/
│   ├── INDEX.md
│   ├── ct_mismatch.md, mri_mismatch.md, cassi_mismatch.md
│   ├── lensless_mismatch.md, microscopy_mismatch.md
│   └── _template.md
├── 03_system_design/
│   ├── INDEX.md
│   ├── ct_system.md, mri_system.md, lensless_system.md, sim_system.md
│   └── _template.md
└── 04_simulation/
    ├── INDEX.md
    ├── ct_simulation.md, optics_simulation.md, mri_simulation.md, wave_simulation.md
    └── _template.md
```

---

## Citation

If you use PWM in your work, please cite:
```
@article{pwm2025,
  title={Physics World Model: A Universal Framework for Computational Imaging},
  author={...},
  year={2025}
}
```

Repository: https://github.com/integritynoble/Physics_World_Model
Live platform: https://pwm.platformai.org
