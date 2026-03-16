# 3D Gaussian Splatting (3DGS) (`gaussian_splatting`)

Category: Neural Rendering

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | EWA Splatting | `pwm_core.recon.gaussian_splatting_solver.run_gaussian_splatting` | No |  |
| `best_quality` | 3DGS (full) | `pwm_core.recon.gaussian_splatting_solver.run_gaussian_splatting` | Yes | Kerbl et al. SIGGRAPH 2023 |
| `famous_dl` | NeRF (baseline comparison) | `pwm_core.recon.nerf_solver.run_nerf` | No |  |
| `small_gpu` | 3DGS (compact) | `pwm_core.recon.gaussian_splatting_solver.run_gaussian_splatting` | No |  |

## Usage

```python
# Import and run
from algorithm_base.gaussian_splatting import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.gaussian_splatting import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| 2DGS | 2024 | 34.0 | — | — |
| Scaffold-GS | 2024 | 33.8 | — | — |
| 3D Gaussian Splatting | 2023 | 33.3 | — | — |
| EWA Splatting (PWM) | — | — | — | — |
| 3DGS (full) (PWM) | — | — | — | — |
| NeRF (baseline comparison) (PWM) | — | — | — | — |
| 3DGS (compact) (PWM) | — | — | — | — |
| direct_render_baseline (test) | — | — | — | — |
