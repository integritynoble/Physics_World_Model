# Single-Pixel Camera (SPC) (`spc`)

Category: Compressive Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | TVAL3 | `pwm_core.recon.cs_solvers.run_tval3` | No |  |
| `best_quality` | ADMM-L1 | `pwm_core.recon.spc_solvers.run_admm_spc` | No | Boyd et al. 2010 |
| `famous_dl` | FISTA-L1 | `pwm_core.recon.spc_solvers.run_fista_l1_spc` | No | Beck & Teboulle 2009 |
| `small_gpu` | FISTA-L1 | `pwm_core.recon.spc_solvers.run_fista_l1_spc` | No |  |
| `ista_net_plus` | ISTA-Net+ | `pwm_core.recon.spc_solvers.run_admm_spc` | No | Zhang & Ghanem, CVPR 2018 |
| `hatnet` | HATNet | `pwm_core.recon.spc_solvers.run_fista_l1_spc` | No | Song et al., TIP 2021 |

## Usage

```python
# Import and run
from algorithm_base.spc import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.spc import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| AMP-Net | 2021 | 34.6 | 27.2 | partial |
| ISTA-Net+ | 2018 | 32.3 | 27.2 | partial |
| TransCS | 2022 | 31.1 | 27.2 | partial |
| CSNet+ | 2019 | 29.8 | 27.2 | done |
| TVAL3 | 2009 | 24.6 | 27.2 | done |
| Random sampling baseline | 2009 | 15.0 | 27.2 | done |
| Pseudoinverse (no regularization) | 2009 | 8.0 | 27.2 | done |
| ADMM-L1 (PWM) | — | 6.8 | 27.2 | done |
