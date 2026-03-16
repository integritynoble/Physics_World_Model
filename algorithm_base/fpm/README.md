# Fourier Ptychographic Microscopy (FPM) (`fpm`)

Category: Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Sequential Phase Retrieval | `pwm_core.recon.fpm_solver.run_fpm` | No |  |
| `best_quality` | Gradient Descent FPM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `famous_dl` | Fourier Ptychnet | `pwm_core.recon.fpm_solver.run_fpm` | No | Jiang et al. 2018, Biomed. Optics Express |
| `small_gpu` | Fourier Ptychnet | `pwm_core.recon.fpm_solver.run_fpm` | No |  |

## Usage

```python
# Import and run
from algorithm_base.fpm import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.fpm import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Gradient descent FPM | 2015 | 30.0 | 30.3 | done |
| GS-FPM | 2013 | 28.0 | 30.3 | done |
| Sequential Phase Retrieval (PWM) | — | 18.2 | 30.3 | done |
| Fourier Ptychnet (PWM) | — | 18.2 | 30.3 | done |
| precomputed_baseline (test) | — | 18.2 | 30.3 | done |
| Single low-res capture | 2013 | 18.0 | 30.3 | done |
