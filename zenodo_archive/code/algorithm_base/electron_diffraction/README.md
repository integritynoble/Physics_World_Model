# 4D-STEM Electron Diffraction (`electron_diffraction`)

Category: Electron Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | ePIE (electron ptychography) | `pwm_core.recon.ptychography_solver.run_epie` | No |  |
| `best_quality` | ED-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `famous_dl` | CRISP-ED [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.electron_diffraction import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.electron_diffraction import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| ED-Net [proxy] (PWM) | — | 44.4 | 22.4 | gap |
| CRISP-ED [proxy] (PWM) | — | 44.4 | 22.4 | gap |
| ePIE (electron ptychography) (PWM) | — | 42.0 | 22.4 | gap |
| precomputed_baseline (test) | — | 42.0 | 22.4 | gap |
| DPC (Differential Phase Contrast) | 2016 | 25.0 | 22.4 | done |
| Center-of-mass analysis | 2014 | 22.0 | 22.4 | done |
