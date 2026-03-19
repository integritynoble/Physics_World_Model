# Lensless (Diffuser Camera) Imaging (`lensless`)

Category: Computational Photography

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | ADMM-TV | `pwm_core.recon.lensless_solver.run_lensless` | No | Antipa et al. 2018 |
| `best_quality` | FlatNet | `pwm_core.recon.flatnet.run_flatnet` | No | Khan et al. TPAMI 2020 |
| `famous_dl` | FlatNet | `pwm_core.recon.flatnet.run_flatnet` | No |  |
| `small_gpu` | FlatNet-Lite | `pwm_core.recon.flatnet.run_flatnet` | No |  |

## Usage

```python
# Import and run
from algorithm_base.lensless import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.lensless import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| LensNet | 2025 | 27.5 | 25.2 | done |
| MWDN | 2023 | 25.7 | 25.2 | done |
| FlatNet | 2022 | 21.2 | 25.2 | done |
| ADMM | 2000 | 12.8 | 25.2 | done |
| FlatNet-Lite (PWM) | — | 11.9 | 25.2 | done |
| wiener_deconv (test) | — | 11.9 | 25.2 | done |
| Wiener deconvolution | 2025 | 7.3 | 25.2 | done |
