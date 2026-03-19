# Neural Radiance Fields (NeRF) (`nerf`)

Category: Neural Rendering

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | SfM + MVS | `pwm_core.recon.nerf_solver.run_nerf` | No |  |
| `best_quality` | Mip-NeRF 360 | `pwm_core.recon.nerf_solver.run_nerf` | Yes | Barron et al. CVPR 2022 |
| `famous_dl` | NeRF (original MLP) | `pwm_core.recon.nerf_solver.run_nerf` | No | Mildenhall et al. 2020 |
| `small_gpu` | Instant-NGP | `pwm_core.recon.nerf_solver.run_nerf` | No | Muller et al. 2022 |
| `rl_proxy` | Richardson-Lucy (proxy baseline) | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `fista_proxy` | FISTA-TV (proxy baseline) | `pwm_core.recon.cs_solvers.run_ista` | No | Beck & Teboulle 2009, SIAM |

## Usage

```python
# Import and run
from algorithm_base.nerf import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.nerf import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Zip-NeRF | 2023 | 33.7 | — | — |
| 3D Gaussian Splatting | 2023 | 33.3 | — | — |
| Instant-NGP | 2022 | 33.2 | — | — |
| TensoRF | 2022 | 33.1 | — | — |
| Mip-NeRF 360 | 2022 | 33.1 | — | — |
| Plenoxels | 2022 | 31.7 | — | — |
| NeRF | 2020 | 31.0 | — | — |
| SfM + MVS (PWM) | — | 29.0 | — | — |
| NeRF (original MLP) (PWM) | — | 29.0 | — | — |
| Richardson-Lucy (proxy baseline) (PWM) | — | 29.0 | — | — |
| FISTA-TV (proxy baseline) (PWM) | — | 29.0 | — | — |
| precomputed_baseline (test) | — | 29.0 | — | — |
