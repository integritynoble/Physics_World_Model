# Scanning Acoustic Microscopy (SAM) (`acoustic_microscopy`)

Category: Industrial Inspection

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `saft_dl` | SAFT-DL [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.acoustic_microscopy import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.acoustic_microscopy import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| SwinIR (SAM) | 2024 | 35.1 | 38.5 | done |
| HDL-SAM (SwinIR+Hypergraph) | 2024 | 31.6 | 38.5 | done |
| Hypergraph Inpainting | 2023 | 28.0 | 38.5 | done |
| SAFT (Synth Aperture Focus) | 1980 | 25.0 | 38.5 | done |
| Adjoint [proxy] (PWM) | — | 24.8 | 38.5 | done |
| PnP-ADMM [proxy] (PWM) | — | 24.8 | 38.5 | done |
| precomputed_baseline (test) | — | 22.6 | 38.5 | done |
| DAS beamforming | 1990 | 22.0 | 38.5 | done |
