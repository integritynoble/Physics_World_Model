# Near-field Scanning Optical Microscopy (NSOM) (`nsom`)

Category: Scanning Probe Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Richardson-Lucy | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No |  |
| `best_quality` | CARE | `pwm_core.recon.care_unet.run_care` | Yes | Weigert et al. 2018 |
| `nsom_dl` | NSOM-Net | `pwm_core.recon.nsom_solvers.nsom_dl_recon` | Yes | Park, J. et al. (2020) DL for near-field optical microscopy, Optica 7(11) |

## Usage

```python
# Import and run
from algorithm_base.nsom import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.nsom import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| BM3D | 2007 | 28.0 | 29.6 | done |
| Deconvolution | 2000 | 24.0 | 29.6 | done |
| Richardson-Lucy (PWM) | — | 24.0 | 29.6 | done |
| CARE (PWM) | — | 24.0 | 29.6 | done |
| NSOM-Net (PWM) | — | 24.0 | 29.6 | done |
| precomputed_baseline (test) | — | 24.0 | 29.6 | done |
