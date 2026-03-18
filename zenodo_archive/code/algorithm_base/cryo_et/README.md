# Cryo-Electron Tomography (Cryo-ET) (`cryo_et`)

Category: Electron Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Richardson-Lucy | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No |  |
| `best_quality` | CARE | `pwm_core.recon.care_unet.run_care` | Yes | Weigert et al. 2018 |
| `cryo_et_dl` | CryoCARE | `pwm_core.recon.cryoet_solvers.cryocare_recon` | Yes | Buchholz, T.O. et al. (2019) Content-aware image restoration for cryo-EM, Methods Enzymol. |

## Usage

```python
# Import and run
from algorithm_base.cryo_et import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.cryo_et import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| IsoNet | 2022 | 28.0 | 30.4 | done |
| SIRT | 1972 | 25.0 | 30.4 | done |
| WBP | 1970 | 22.0 | 30.4 | done |
| Richardson-Lucy (PWM) | — | 13.2 | 30.4 | done |
| CARE (PWM) | — | 13.2 | 30.4 | done |
| CryoCARE (PWM) | — | 13.2 | 30.4 | done |
| precomputed_baseline (test) | — | 13.2 | 30.4 | done |
| WBP (45-deg missing wedge) | 2019 | 13.1 | 30.4 | done |
