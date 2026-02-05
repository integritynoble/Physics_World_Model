# PWM CasePacks - 18 Imaging Modalities

CasePacks are validated templates that define modality-specific configurations for simulation, reconstruction, and analysis.

## Overview

| # | Modality | CasePack File | Solver | Benchmark PSNR |
|---|----------|---------------|--------|----------------|
| 1 | Widefield | `widefield_deconv_basic_v1.json` | Richardson-Lucy | 27.31 dB |
| 2 | Widefield Low-Dose | `widefield_lowdose_highbg_v1.json` | PnP | 27.78 dB |
| 3 | Confocal Live-Cell | `confocal_livecell_lowdose_drift_v1.json` | Richardson-Lucy | 26.27 dB |
| 4 | Confocal 3D | `confocal_3d_stack_attenuation_v1.json` | 3D Richardson-Lucy | 29.01 dB |
| 5 | SIM | `sim_3x3_fragile_v1.json` | Wiener | 27.48 dB |
| 6 | CASSI | `cassi_spectral_imaging_v1.json` | GAP + HSI_SDeCNN | 30.60 dB |
| 7 | SPC | `spc_low_sampling_poisson_v1.json` | PnP-FISTA + DRUNet | 30.90 dB |
| 8 | CACTI | `cacti_video_sci_v1.json` | GAP-TV | 32.79 dB |
| 9 | Lensless | `lensless_diffusercam_basic_v1.json` | ADMM-TV | 34.66 dB |
| 10 | Light-Sheet | `lightsheet_tissue_stripes_scatter_v1.json` | Stripe Removal | 28.05 dB |
| 11 | CT | `ct_conebeam_lowdose_scatter_v1.json` | PnP-SART + DRUNet | 27.97 dB |
| 12 | MRI | `mri_cartesian_accelerated_v1.json` | PnP-ADMM + DRUNet | 48.25 dB |
| 13 | Ptychography | `ptychography_phase_retrieval_v1.json` | Neural Network | 59.47 dB |
| 14 | Holography | `holography_offaxis_phase_v1.json` | Neural Network | 42.52 dB |
| 15 | NeRF | `nerf_from_poses_basic_v1.json` | Neural Implicit (SIREN) | 61.35 dB |
| 16 | 3D Gaussian Splatting | `gaussian_splatting_basic_v1.json` | 2D Gaussian Opt | 30.47 dB |
| 17 | Matrix (Generic) | `matrix_generic_linear_v1.json` | FISTA-TV | 25.79 dB |
| 18 | Panorama Multifocal | `panorama_multifocal_fusion_v1.json` | Neural Fusion | 27.78 dB |

## CasePack Categories

### Microscopy
- **Widefield** - Classical PSF deconvolution with Richardson-Lucy
- **Widefield Low-Dose** - Low photon count imaging with PnP denoising
- **Confocal Live-Cell** - Live-cell imaging with motion/drift handling
- **Confocal 3D** - 3D stack deconvolution with axial PSF elongation
- **SIM** - Structured Illumination Microscopy (2x resolution)
- **Light-Sheet** - Stripe artifact removal and multi-view fusion

### Compressive Imaging
- **SPC** - Single-Pixel Camera with Hadamard patterns
- **CASSI** - Coded Aperture Snapshot Spectral Imaging (hyperspectral)
- **CACTI** - Coded Aperture Compressive Temporal Imaging (video)
- **Lensless** - DiffuserCam lensless imaging

### Medical Imaging
- **CT** - Computed Tomography (FBP, SART, PnP)
- **MRI** - Magnetic Resonance Imaging (parallel imaging, CS)

### Coherent Imaging
- **Ptychography** - Phase retrieval from diffraction patterns
- **Holography** - Off-axis digital holography

### Neural Rendering
- **NeRF** - Neural Radiance Fields for novel view synthesis
- **3D Gaussian Splatting** - Differentiable Gaussian splatting

### General
- **Matrix** - Generic linear inverse problem (y = Ax)
- **Panorama Multifocal** - Multi-view focus stacking

## Usage

### CLI

```bash
# Run with a specific casepack
pwm run --casepack widefield_deconv_basic_v1

# Run with custom input
pwm run --casepack spc_low_sampling_poisson_v1 --input my_image.png

# Calibrate and reconstruct
pwm calib-recon --casepack cassi_measured_y_fit_theta_v1 --y measured.npy
```

### Python API

```python
from pwm_core.core.registry import load_casepack
from pwm_core.core.runner import run_from_casepack

# Load casepack
casepack = load_casepack("widefield_deconv_basic_v1")

# Run simulation + reconstruction
result = run_from_casepack(casepack)

# Access results
print(f"PSNR: {result.metrics['psnr']:.2f} dB")
```

## CasePack Structure

Each casepack JSON file contains:

```json
{
  "casepack_version": "0.2.1",
  "id": "unique_casepack_id",
  "title": "Human-readable title",
  "modality": "modality_name",
  "tags": ["tag1", "tag2"],
  "keywords": ["keyword1", "keyword2"],
  "required_user_inputs": [],
  "base_spec": {
    "version": "0.2.1",
    "states": {
      "physics": { "modality": "..." },
      "budget": { ... },
      "calibration": { ... },
      "sensor": { ... },
      "task": { ... }
    },
    "recon": {
      "portfolio": {
        "solvers": [...]
      }
    }
  },
  "overrides": { ... },
  "notes": "Description and usage notes",
  "benchmark_results": {
    "psnr_db": 30.0,
    "reference_psnr_db": 28.0,
    "status": "pass"
  }
}
```

## Adding a New CasePack

1. Copy an existing casepack as a template
2. Update the `id`, `title`, `modality`, and `tags`
3. Configure `base_spec.states` for your modality
4. Add appropriate solvers to `recon.portfolio`
5. Set `overrides` for modality-specific parameters
6. Run benchmark to validate performance

See `contrib/templates/` for operator and calibrator templates.

## Operator-Fit CasePacks

For real measurements with imperfect forward models:

- `cassi_measured_y_fit_theta_v1.json` - CASSI dispersion/mask calibration
- `generic_matrix_yA_fit_gain_shift_v1.json` - Generic gain/shift calibration

These casepacks include calibration loops to fit operator parameters before reconstruction.

## Benchmark Validation

All casepacks are validated against reference implementations:

```bash
# Run all 18 modality benchmarks
python -m packages.pwm_core.benchmarks.run_all --all

# Run specific modality
python -m packages.pwm_core.benchmarks.run_all --modality mri
```

Results are saved to `benchmarks/results/benchmark_report.md`.

---

*Last updated: 2026-02-04*
