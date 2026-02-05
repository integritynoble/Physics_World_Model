# PWM Usage Guide

This guide covers the three main workflows in PWM:

1. **Prompt-run** - Run simulations from natural language prompts
2. **Operator-correction** - Calibrate operators from measured data
3. **RunBundle+Viewer** - View and analyze results

---

## Prerequisites

### Installation

```bash
# Install pwm_core in editable mode
pip install -e packages/pwm_core

# Install with viewer support (Streamlit)
pip install -e "packages/pwm_core[viewer]"

# Verify installation
pwm --help
```

### Available Commands

```
pwm run           # Run full pipeline from prompt or spec
pwm fit-operator  # Fit operator parameters from measured data
pwm calib-recon   # Calibrate operator and reconstruct
pwm view          # Launch interactive viewer
```

---

## 1. Prompt-Run Workflow

Run simulations using natural language prompts. PWM automatically selects the appropriate CasePack and runs the full pipeline.

### Basic Usage

```bash
# Run with a prompt
pwm run --prompt "widefield microscopy deconvolution"

# Specify output directory
pwm run --prompt "confocal live-cell low dose" --out-dir my_runs
```

### Run from Spec File

```bash
# Run from a JSON spec file
pwm run --spec examples/specs/widefield_basic_spec.json --out-dir runs
```

### Output

The command outputs a JSON summary:

```json
{
  "spec_id": "widefield_deconv_basic_v1_base",
  "runbundle_path": "runs/run_widefield_deconv_basic_v1_base_abc123",
  "diagnosis": {
    "verdict": "Dose Limited",
    "confidence": 0.7,
    "bottleneck": "Dose Limited"
  },
  "recon": [
    {
      "solver_id": "tv_fista",
      "metrics": {
        "mse": 0.0098,
        "psnr": 18.7
      }
    }
  ]
}
```

---

## 2. Operator-Correction Workflow

Calibrate forward operators from measured data. Use this when you have real measurements and an imperfect forward model.

### fit-operator: Fit Parameters Only

```bash
pwm fit-operator --y examples/data/widefield_basic/measurements.npy --operator widefield --out-dir runs
```

### calib-recon: Calibrate and Reconstruct

```bash
pwm calib-recon --y examples/data/widefield_basic/measurements.npy --operator widefield --out-dir runs
```

### Supported Operators

| Operator ID | Description |
|-------------|-------------|
| `widefield` | Widefield microscopy (Gaussian PSF blur) |
| `confocal` | Confocal microscopy |
| `sim` | Structured illumination microscopy |
| `cassi` | Coded aperture spectral imaging |
| `spc` | Single-pixel camera |
| `cacti` | CACTI video snapshot compressive imaging |
| `lensless` | Lensless imaging (diffuser) |
| `lightsheet` | Light-sheet microscopy |
| `ct` | Computed tomography |
| `mri` | Magnetic resonance imaging |
| `ptychography` | Ptychographic imaging |
| `holography` | Holographic imaging |
| `nerf` | Neural radiance fields (multi-view) |
| `gaussian_splatting` | 3D Gaussian splatting |
| `matrix` | Generic matrix operator |

---

## 3. RunBundle + Viewer

Every run creates a RunBundle - a self-contained directory with all artifacts for reproducibility.

### RunBundle Structure

```
run_<spec_id>_<hash>/
├── artifacts/
│   ├── images/           # PNG visualizations
│   │   ├── x_hat.png     # Reconstruction
│   │   ├── x_true.png    # Ground truth
│   │   ├── y.png         # Measurement
│   │   └── comparison.png
│   ├── x_hat.npy         # Reconstruction array
│   ├── x_true.npy        # Ground truth array
│   ├── y.npy             # Measurement array
│   └── metrics.json      # Quality metrics
├── internal_state/
│   ├── diagnosis.json    # Bottleneck analysis
│   └── recon_info.json   # Solver details
├── logs/
└── provenance.json       # Environment info
```

### Launch Viewer

```bash
# View a specific RunBundle
pwm view runs/run_widefield_deconv_basic_v1_base_abc123
```

---

## 4. Complete Modality Examples

This section provides examples for all 16 imaging modalities supported by PWM.

---

### 4.1 Widefield Microscopy (Basic)

Basic widefield deconvolution with Gaussian PSF.

**Prompt-Run:**
```bash
pwm run --prompt "widefield microscopy deconvolution" --out-dir runs
```

**Spec File:** `examples/specs/widefield_basic_spec.json`
```json
{
  "version": "0.2.1",
  "id": "widefield_basic_example",
  "input": {
    "mode": "simulate"
  },
  "states": {
    "physics": {
      "modality": "widefield"
    },
    "budget": {
      "photon_budget": {"max_photons": 1000}
    },
    "sensor": {
      "shot_noise": {"enabled": true, "model": "poisson"},
      "read_noise_sigma": 5.0
    },
    "task": {
      "kind": "simulate_recon_analyze"
    }
  },
  "recon": {
    "portfolio": {
      "solvers": [{"id": "tv_fista", "params": {"lam": 0.02, "iters": 200}}]
    }
  }
}
```

**Operator-Correction:**
```bash
pwm calib-recon --y examples/data/widefield_basic/measurements.npy --operator widefield --out-dir runs
```

**Measurement Data:** `examples/data/widefield_basic/measurements.npy` (64x64 blurred + noisy image)

---

### 4.2 Widefield Microscopy (Low Dose)

Low photon count imaging with high background.

**Prompt-Run:**
```bash
pwm run --prompt "widefield low dose high background" --out-dir runs
```

**Spec File:** `examples/specs/widefield_lowdose_spec.json`
```json
{
  "version": "0.2.1",
  "id": "widefield_lowdose_example",
  "input": {
    "mode": "simulate"
  },
  "states": {
    "physics": {
      "modality": "widefield"
    },
    "budget": {
      "photon_budget": {"max_photons": 100}
    },
    "sensor": {
      "shot_noise": {"enabled": true, "model": "poisson"},
      "read_noise_sigma": 10.0
    },
    "task": {
      "kind": "simulate_recon_analyze"
    }
  }
}
```

**Operator-Correction:**
```bash
pwm calib-recon --y examples/data/widefield_lowdose/measurements.npy --operator widefield --out-dir runs
```

**Measurement Data:** `examples/data/widefield_lowdose/measurements.npy`

---

### 4.3 Confocal Live-Cell

Confocal microscopy with sample drift.

**Prompt-Run:**
```bash
pwm run --prompt "confocal live-cell low dose drift" --out-dir runs
```

**Spec File:** `examples/specs/confocal_livecell_spec.json`
```json
{
  "version": "0.2.1",
  "id": "confocal_livecell_example",
  "input": {
    "mode": "simulate"
  },
  "states": {
    "physics": {
      "modality": "confocal"
    },
    "budget": {
      "photon_budget": {"max_photons": 500}
    },
    "sample": {
      "motion": {"type": "drift", "rate_px_per_frame": 0.1}
    },
    "sensor": {
      "shot_noise": {"enabled": true, "model": "poisson"},
      "read_noise_sigma": 3.0
    },
    "task": {
      "kind": "simulate_recon_analyze"
    }
  }
}
```

**Operator-Correction:**
```bash
pwm calib-recon --y examples/data/confocal_livecell/measurements.npy --operator confocal --out-dir runs
```

**Measurement Data:** `examples/data/confocal_livecell/measurements.npy`

---

### 4.4 Confocal 3D Stack

3D confocal imaging with depth attenuation.

**Prompt-Run:**
```bash
pwm run --prompt "confocal 3d stack attenuation" --out-dir runs
```

**Spec File:** `examples/specs/confocal_3d_spec.json`
```json
{
  "version": "0.2.1",
  "id": "confocal_3d_example",
  "input": {
    "mode": "simulate"
  },
  "states": {
    "physics": {
      "modality": "confocal",
      "dims": {"H": 64, "W": 64, "D": 32}
    },
    "environment": {
      "attenuation": {"model": "exponential", "coefficient": 0.03}
    },
    "sensor": {
      "shot_noise": {"enabled": true, "model": "poisson"},
      "read_noise_sigma": 5.0
    },
    "task": {
      "kind": "simulate_recon_analyze"
    }
  }
}
```

**Operator-Correction:**
```bash
pwm calib-recon --y examples/data/confocal_3d/measurements.npy --operator confocal --out-dir runs
```

**Measurement Data:** `examples/data/confocal_3d/measurements.npy` (64x64x32 3D stack)

---

### 4.5 Structured Illumination Microscopy (SIM)

SIM with 3x3 pattern acquisition.

**Prompt-Run:**
```bash
pwm run --prompt "SIM structured illumination 9 frames" --out-dir runs
```

**Spec File:** `examples/specs/sim_spec.json`
```json
{
  "version": "0.2.1",
  "id": "sim_example",
  "input": {
    "mode": "simulate"
  },
  "states": {
    "physics": {
      "modality": "sim"
    },
    "budget": {
      "photon_budget": {"max_photons": 600},
      "measurement_budget": {"phases": 3, "angles": 3}
    },
    "calibration": {
      "pattern": {"freq_px": 8.0, "mod_depth": 0.8}
    },
    "sensor": {
      "shot_noise": {"enabled": true, "model": "poisson"},
      "read_noise_sigma": 3.0
    },
    "task": {
      "kind": "simulate_recon_analyze"
    }
  }
}
```

**Operator-Correction:**
```bash
pwm calib-recon --y examples/data/sim/measurements.npy --operator sim --out-dir runs
```

**Measurement Data:** `examples/data/sim/measurements.npy` (64x64x9 pattern stack)

---

### 4.6 CASSI Spectral Imaging

Coded aperture snapshot spectral imaging.

**Prompt-Run:**
```bash
pwm run --prompt "CASSI spectral coded aperture" --out-dir runs
```

**Spec File:** `examples/specs/cassi_spec.json`
```json
{
  "version": "0.2.1",
  "id": "cassi_example",
  "input": {
    "mode": "simulate"
  },
  "states": {
    "physics": {
      "modality": "cassi",
      "dims": {"H": 64, "W": 64, "L": 16}
    },
    "budget": {
      "photon_budget": {"max_photons": 800}
    },
    "sensor": {
      "shot_noise": {"enabled": true, "model": "poisson"},
      "read_noise_sigma": 5.0
    },
    "task": {
      "kind": "simulate_recon_analyze"
    }
  }
}
```

**Operator-Correction:**
```bash
pwm calib-recon --y examples/data/cassi/measurements.npy --operator cassi --out-dir runs
```

**Measurement Data:** `examples/data/cassi/measurements.npy` (64x64 coded snapshot)
**Ground Truth:** `examples/data/cassi/ground_truth.npy` (64x64x16 spectral cube)

---

### 4.7 Single-Pixel Camera (SPC)

Compressed sensing single-pixel imaging.

**Prompt-Run:**
```bash
pwm run --prompt "single-pixel camera compressed sensing" --out-dir runs
```

**Spec File:** `examples/specs/spc_spec.json`
```json
{
  "version": "0.2.1",
  "id": "spc_example",
  "input": {
    "mode": "simulate"
  },
  "states": {
    "physics": {
      "modality": "spc",
      "dims": {"H": 64, "W": 64}
    },
    "budget": {
      "measurement_budget": {"sampling_rate": 0.15}
    },
    "sensor": {
      "shot_noise": {"enabled": true, "model": "poisson"},
      "read_noise_sigma": 10.0
    },
    "task": {
      "kind": "simulate_recon_analyze"
    }
  }
}
```

**Operator-Correction:**
```bash
pwm calib-recon --y examples/data/spc/measurements.npy --operator spc --out-dir runs
```

**Measurement Data:** `examples/data/spc/measurements.npy` (614 compressed measurements)

---

### 4.8 CACTI (Video Snapshot Compressive Imaging)

Coded Aperture Compressive Temporal Imaging for video compression.

**Prompt-Run:**
```bash
pwm run --prompt "CACTI video snapshot compressive imaging" --out-dir runs
```

**Spec File:** `examples/specs/cacti_spec.json`
```json
{
  "version": "0.2.1",
  "id": "cacti_example",
  "input": {
    "mode": "simulate"
  },
  "states": {
    "physics": {
      "modality": "cacti",
      "dims": {"H": 64, "W": 64, "T": 8}
    },
    "budget": {
      "photon_budget": {"max_photons": 800}
    },
    "sensor": {
      "shot_noise": {"enabled": true, "model": "poisson"},
      "read_noise_sigma": 5.0
    },
    "task": {
      "kind": "simulate_recon_analyze"
    }
  }
}
```

**Operator-Correction:**
```bash
pwm calib-recon --y examples/data/cacti/measurements.npy --operator cacti --out-dir runs
```

**Measurement Data:** `examples/data/cacti/measurements.npy` (64x64 snapshot from 8-frame video)
**Ground Truth:** `examples/data/cacti/ground_truth.npy` (64x64x8 video cube)

---

### 4.9 Lensless (Diffuser) Imaging

Lensless imaging with diffuser PSF.

**Prompt-Run:**
```bash
pwm run --prompt "lensless diffuser camera" --out-dir runs
```

**Spec File:** `examples/specs/lensless_spec.json`
```json
{
  "version": "0.2.1",
  "id": "lensless_example",
  "input": {
    "mode": "simulate"
  },
  "states": {
    "physics": {
      "modality": "lensless"
    },
    "budget": {
      "photon_budget": {"max_photons": 500}
    },
    "sensor": {
      "shot_noise": {"enabled": true, "model": "poisson"},
      "read_noise_sigma": 8.0
    },
    "task": {
      "kind": "simulate_recon_analyze"
    }
  }
}
```

**Operator-Correction:**
```bash
pwm calib-recon --y examples/data/lensless/measurements.npy --operator lensless --out-dir runs
```

**Measurement Data:** `examples/data/lensless/measurements.npy` (64x64 sensor capture)

---

### 4.9 Light-Sheet Microscopy

Light-sheet imaging with stripe artifacts.

**Prompt-Run:**
```bash
pwm run --prompt "lightsheet tissue stripes scatter" --out-dir runs
```

**Spec File:** `examples/specs/lightsheet_spec.json`
```json
{
  "version": "0.2.1",
  "id": "lightsheet_example",
  "input": {
    "mode": "simulate"
  },
  "states": {
    "physics": {
      "modality": "lightsheet",
      "dims": {"H": 64, "W": 64, "D": 32}
    },
    "environment": {
      "stripe_artifacts": {"enabled": true, "strength": 0.2},
      "scatter": {"enabled": true, "strength": 0.1}
    },
    "sensor": {
      "shot_noise": {"enabled": true, "model": "poisson"},
      "read_noise_sigma": 4.0
    },
    "task": {
      "kind": "simulate_recon_analyze"
    }
  }
}
```

**Operator-Correction:**
```bash
pwm calib-recon --y examples/data/lightsheet/measurements.npy --operator lightsheet --out-dir runs
```

**Measurement Data:** `examples/data/lightsheet/measurements.npy` (64x64x32 with stripes)

---

### 4.10 Cone-Beam CT

Low-dose CT with scatter.

**Prompt-Run:**
```bash
pwm run --prompt "CT cone-beam low dose scatter" --out-dir runs
```

**Spec File:** `examples/specs/ct_spec.json`
```json
{
  "version": "0.2.1",
  "id": "ct_example",
  "input": {
    "mode": "simulate"
  },
  "states": {
    "physics": {
      "modality": "ct"
    },
    "budget": {
      "photon_budget": {"max_photons": 500},
      "measurement_budget": {"views": 180}
    },
    "environment": {
      "scatter": {"enabled": true, "strength": 0.15}
    },
    "sensor": {
      "shot_noise": {"enabled": true, "model": "poisson"},
      "read_noise_sigma": 5.0
    },
    "task": {
      "kind": "simulate_recon_analyze"
    }
  }
}
```

**Operator-Correction:**
```bash
pwm calib-recon --y examples/data/ct/measurements.npy --operator ct --out-dir runs
```

**Measurement Data:** `examples/data/ct/measurements.npy` (180x64 sinogram)

---

### 4.11 MRI (Accelerated)

Accelerated MRI with k-space undersampling.

**Prompt-Run:**
```bash
pwm run --prompt "MRI accelerated k-space undersampling" --out-dir runs
```

**Spec File:** `examples/specs/mri_spec.json`
```json
{
  "version": "0.2.1",
  "id": "mri_example",
  "input": {
    "mode": "simulate"
  },
  "states": {
    "physics": {
      "modality": "mri"
    },
    "budget": {
      "measurement_budget": {"sampling_rate": 0.25}
    },
    "calibration": {
      "coils": 8
    },
    "sensor": {
      "shot_noise": {"enabled": false},
      "read_noise_sigma": 0.005
    },
    "task": {
      "kind": "simulate_recon_analyze"
    }
  }
}
```

**Operator-Correction:**
```bash
pwm calib-recon --y examples/data/mri/measurements.npy --operator mri --out-dir runs
```

**Measurement Data:** `examples/data/mri/measurements.npy` (64x64 complex k-space)

---

### 4.12 Ptychography

Ptychographic phase retrieval.

**Prompt-Run:**
```bash
pwm run --prompt "ptychography phase retrieval" --out-dir runs
```

**Spec File:** `examples/specs/ptychography_spec.json`
```json
{
  "version": "0.2.1",
  "id": "ptychography_example",
  "input": {
    "mode": "simulate"
  },
  "states": {
    "physics": {
      "modality": "ptychography"
    },
    "budget": {
      "measurement_budget": {"scan_positions": 16}
    },
    "sensor": {
      "shot_noise": {"enabled": true, "model": "poisson"},
      "read_noise_sigma": 5.0
    },
    "task": {
      "kind": "simulate_recon_analyze"
    }
  }
}
```

**Operator-Correction:**
```bash
pwm calib-recon --y examples/data/ptychography/measurements.npy --operator ptychography --out-dir runs
```

**Measurement Data:** `examples/data/ptychography/measurements.npy` (16x64x64 diffraction patterns)

---

### 4.13 Holography

Off-axis digital holography.

**Prompt-Run:**
```bash
pwm run --prompt "holography off-axis phase" --out-dir runs
```

**Spec File:** `examples/specs/holography_spec.json`
```json
{
  "version": "0.2.1",
  "id": "holography_example",
  "input": {
    "mode": "simulate"
  },
  "states": {
    "physics": {
      "modality": "holography"
    },
    "budget": {
      "photon_budget": {"max_photons": 1000}
    },
    "sensor": {
      "shot_noise": {"enabled": true, "model": "poisson"},
      "read_noise_sigma": 3.0
    },
    "task": {
      "kind": "simulate_recon_analyze"
    }
  }
}
```

**Operator-Correction:**
```bash
pwm calib-recon --y examples/data/holography/measurements.npy --operator holography --out-dir runs
```

**Measurement Data:** `examples/data/holography/measurements.npy` (64x64 hologram)

---

### 4.14 NeRF (Neural Radiance Fields)

Neural radiance field from multi-view images.

**Prompt-Run:**
```bash
pwm run --prompt "nerf 3d rendering multi-view" --out-dir runs
```

**Spec File:** `examples/specs/nerf_spec.json`
```json
{
  "version": "0.2.1",
  "id": "nerf_example",
  "input": {
    "mode": "simulate"
  },
  "states": {
    "physics": {
      "modality": "nerf",
      "rendering": {"n_views": 10}
    },
    "sensor": {
      "shot_noise": {"enabled": true, "model": "poisson"},
      "read_noise_sigma": 5.0
    },
    "task": {
      "kind": "simulate_recon_analyze"
    }
  }
}
```

**Operator-Correction:**
```bash
pwm calib-recon --y examples/data/nerf/measurements.npy --operator nerf --out-dir runs
```

**Measurement Data:** `examples/data/nerf/measurements.npy` (10x64x64x3 multi-view RGB)

---

### 4.15 3D Gaussian Splatting

3D Gaussian splatting from multi-view images.

**Prompt-Run:**
```bash
pwm run --prompt "gaussian splatting 3dgs multi-view" --out-dir runs
```

**Spec File:** `examples/specs/gaussian_splatting_spec.json`
```json
{
  "version": "0.2.1",
  "id": "gaussian_splatting_example",
  "input": {
    "mode": "simulate"
  },
  "states": {
    "physics": {
      "modality": "gaussian_splatting",
      "rendering": {"n_views": 10}
    },
    "sensor": {
      "shot_noise": {"enabled": true, "model": "poisson"},
      "read_noise_sigma": 5.0
    },
    "task": {
      "kind": "simulate_recon_analyze"
    }
  }
}
```

**Operator-Correction:**
```bash
pwm calib-recon --y examples/data/gaussian_splatting/measurements.npy --operator gaussian_splatting --out-dir runs
```

**Measurement Data:** `examples/data/gaussian_splatting/measurements.npy` (10x64x64x3 multi-view RGB)

---

### 4.16 Generic Matrix (y = Ax)

Generic linear inverse problem with explicit matrix.

**Prompt-Run:**
```bash
pwm run --prompt "matrix linear inverse problem" --out-dir runs
```

**Spec File:** `examples/specs/matrix_generic_spec.json`
```json
{
  "version": "0.2.1",
  "id": "matrix_generic_example",
  "input": {
    "mode": "simulate"
  },
  "states": {
    "physics": {
      "modality": "matrix"
    },
    "budget": {
      "photon_budget": {"max_photons": 1000}
    },
    "sensor": {
      "shot_noise": {"enabled": true, "model": "poisson"},
      "read_noise_sigma": 5.0
    },
    "task": {
      "kind": "simulate_recon_analyze"
    }
  }
}
```

**Operator-Correction:**
```bash
pwm calib-recon --y examples/data/matrix_generic/measurements.npy --operator matrix --out-dir runs
```

**Measurement Data:**
- `examples/data/matrix_generic/measurements.npy` (256 measurements)
- `examples/data/matrix_generic/forward_matrix.npy` (256x4096 matrix A)

---

## Python API

For programmatic use, import the endpoints directly:

```python
from pwm_core.api import endpoints

# Prompt-run
result = endpoints.run(prompt="widefield deconvolution", out_dir="runs")

# From spec dict
spec = {
    "version": "0.2.1",
    "id": "my_experiment",
    "input": {"mode": "simulate"},
    "states": {
        "physics": {"modality": "widefield"},
        "task": {"kind": "simulate_recon_analyze"}
    }
}
result = endpoints.run(spec=spec, out_dir="runs")

# Compile prompt to spec (without running)
compile_result = endpoints.compile_prompt("SIM live-cell low dose")
print(f"Selected CasePack: {compile_result.casepack_id}")

# Launch viewer
endpoints.view("runs/run_widefield_abc123")
```

---

## Example Scripts

The `examples/` directory contains complete workflow examples:

### prompt_to_casepack.py

Full prompt-to-view workflow:

```bash
python examples/prompt_to_casepack.py --prompt "SIM live-cell low dose"
```

### generate_example_data.py

Generate synthetic data for all modalities:

```bash
python examples/generate_example_data.py
```

---

## Troubleshooting

### Common Issues

**1. "No CasePack matched"**
- Try more specific keywords in your prompt
- Check available CasePacks: `ls packages/pwm_core/contrib/casepacks/`

**2. "Measurement file not found"**
- Ensure the path to `--y` file is correct
- Supported formats: `.npy`, `.npz`, `.pt`

**3. "Streamlit not installed"**
- Install viewer extra: `pip install -e "packages/pwm_core[viewer]"`

**4. "CUDA out of memory"**
- Set in spec: `"compute": {"device": "cpu"}`

---

## Quick Reference

| Task | Command |
|------|---------|
| Run from prompt | `pwm run --prompt "..."` |
| Run from spec | `pwm run --spec spec.json` |
| Fit operator only | `pwm fit-operator --y y.npy --operator <id>` |
| Calibrate + reconstruct | `pwm calib-recon --y y.npy --operator <id>` |
| View results | `pwm view runs/<runbundle>` |
| Generate example data | `python examples/generate_example_data.py` |
