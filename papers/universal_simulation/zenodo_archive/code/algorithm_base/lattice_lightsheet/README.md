# Lattice Light-Sheet Microscopy (`lattice_lightsheet`)

Category: Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Richardson-Lucy | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No |  |
| `best_quality` | CARE | `pwm_core.recon.care_unet.run_care` | Yes | Weigert et al. 2018 |
| `llsm_dl` | LLSM-CARE | `pwm_core.recon.llsm_solvers.llsm_care_recon` | Yes | Weigert, M. et al. (2018) Content-aware restoration for lattice light-sheet, Nature Methods 15:1090 |

## Usage

```python
# Import and run
from algorithm_base.lattice_lightsheet import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.lattice_lightsheet import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| CARE 3D | 2018 | 32.0 | 34.0 | done |
| Richardson-Lucy 3D | 1972 | 26.0 | 34.0 | done |
| LLSM-CARE (PWM) | — | 25.1 | 34.0 | done |
| precomputed_baseline (test) | — | 25.1 | 34.0 | done |
