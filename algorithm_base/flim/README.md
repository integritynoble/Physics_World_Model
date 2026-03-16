# Fluorescence Lifetime Imaging (FLIM) (`flim`)

Category: Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Phasor Analysis | `pwm_core.recon.flim_solver.run_flim` | No |  |
| `best_quality` | MLE Fit | `pwm_core.recon.flim_solver.run_flim` | No | Becker 2012, J. Microscopy |
| `famous_dl` | MLE Fit (iterative) | `pwm_core.recon.flim_solver.run_flim` | No | Becker 2012, J. Microscopy |
| `small_gpu` | Phasor Analysis | `pwm_core.recon.flim_solver.run_flim` | No | Digman et al. 2008, Biophysical Journal |

## Usage

```python
# Import and run
from algorithm_base.flim import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.flim import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| MLE Fit (PWM) | — | 36.9 | 35.5 | done |
| MLE Fit (iterative) (PWM) | — | 36.9 | 35.5 | done |
| precomputed_baseline (test) | — | 36.9 | 35.5 | done |
| Net-FLIM (DL) | 2019 | 30.0 | 35.5 | done |
| Phasor approach | 2008 | 25.0 | 35.5 | done |
| Multi-exponential fitting | 2000 | 22.0 | 35.5 | done |
