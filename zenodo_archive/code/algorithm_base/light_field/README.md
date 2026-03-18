# Light Field Imaging (`light_field`)

Category: Computational Optics

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Shift-and-Sum | `pwm_core.recon.light_field_solver.run_light_field` | No |  |
| `best_quality` | LFBM5D | `pwm_core.recon.light_field_solver.lfbm5d_recon` | No | Alain et al. 2017, Signal Processing: Image Communication |
| `famous_dl` | LFSSR | `pwm_core.recon.light_field_solver.lfssr_recon` | No | Yeung et al. ECCV 2018 |
| `small_gpu` | LFSSR | `pwm_core.recon.light_field_solver.lfssr_recon` | No |  |

## Usage

```python
# Import and run
from algorithm_base.light_field import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.light_field import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| DistgSSR | 2021 | 34.8 | 38.2 | done |
| LFT | 2022 | 34.8 | 38.2 | done |
| EPIT | 2022 | 34.8 | 38.2 | done |
| LF-InterNet | 2020 | 34.1 | 38.2 | done |
| LFSSR | 2018 | 33.7 | 38.2 | done |
| DistgEPIT | 2023 | 30.7 | 38.2 | done |
| VDSR (4x SR) | 2016 | 28.6 | 38.2 | done |
| Shift-and-Sum (PWM) | — | 27.3 | 38.2 | done |
| LFBM5D (PWM) | — | 27.3 | 38.2 | done |
| precomputed_baseline (test) | — | 27.3 | 38.2 | done |
| Bicubic (4x SR) | 2019 | 26.5 | 38.2 | done |
