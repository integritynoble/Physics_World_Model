# Focused Ion Beam SEM (FIB-SEM) (`fib_sem`)

Category: Electron Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Richardson-Lucy | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No |  |
| `best_quality` | CARE | `pwm_core.recon.care_unet.run_care` | Yes | Weigert et al. 2018 |
| `fibsem_dl` | FIB-SEM-Net | `pwm_core.recon.fibsem_solvers.fibsem_dl_recon` | Yes | Heinrich, L. et al. (2021) Whole-cell organelle segmentation in volume EM, Nature 599:141 |

## Usage

```python
# Import and run
from algorithm_base.fib_sem import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.fib_sem import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| SwinIR | 2021 | 34.0 | 25.4 | partial |
| NRRN | 2021 | 31.0 | 25.4 | partial |
| BM3D | 2007 | 30.0 | 25.4 | partial |
| Richardson-Lucy (PWM) | — | 28.3 | 25.4 | done |
| CARE (PWM) | — | 28.3 | 25.4 | done |
| FIB-SEM-Net (PWM) | — | 28.3 | 25.4 | done |
| precomputed_baseline (test) | — | 28.3 | 25.4 | done |
