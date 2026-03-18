# Electron Backscatter Diffraction (EBSD) (`ebsd`)

Category: Electron Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FISTA-L2 (Hough baseline) | `pwm_core.recon.classical.run_fista_l2` | No |  |
| `best_quality` | EBSD-DL (DictIndex) [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `famous_dl` | EMsoft-EBSD [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.ebsd import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.ebsd import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| EBSD-DL (DictIndex) [proxy] (PWM) | — | 34.8 | 27.3 | partial |
| EMsoft-EBSD [proxy] (PWM) | — | 34.8 | 27.3 | partial |
| Dictionary indexing | 2015 | 25.0 | 27.3 | done |
| Hough indexing | 1992 | 22.0 | 27.3 | done |
| precomputed_baseline (test) | — | 21.9 | 27.3 | done |
