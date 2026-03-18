# High Dynamic Range (HDR) Imaging (`hdr_imaging`)

Category: Computational Photography

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `hdr_dl` | HDR-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.hdr_imaging import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.hdr_imaging import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| HDR-Transformer | 2022 | 42.4 | 22.9 | gap |
| AHDRNet | 2019 | 41.1 | 22.9 | gap |
| Adjoint [proxy] (PWM) | — | 40.5 | 22.9 | gap |
| PnP-ADMM [proxy] (PWM) | — | 40.5 | 22.9 | gap |
| precomputed_baseline (test) | — | 38.6 | 22.9 | gap |
| Debevec | 1997 | 30.0 | 22.9 | partial |
