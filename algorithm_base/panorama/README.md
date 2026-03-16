# Panorama Multi-Focus Fusion (`panorama`)

Category: Computational Photography

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Laplacian Pyramid Fusion | `pwm_core.recon.panorama_solver.run_panorama_fusion` | No |  |
| `best_quality` | Guided Filter Fusion | `pwm_core.recon.panorama_solver.run_panorama_fusion` | No |  |
| `famous_dl` | IFCNN | `pwm_core.recon.ifcnn.run_ifcnn` | No | Zhang et al. 2020 |
| `small_gpu` | IFCNN | `pwm_core.recon.ifcnn.run_ifcnn` | No |  |

## Usage

```python
# Import and run
from algorithm_base.panorama import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.panorama import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Deep homography | 2023 | 33.6 | 28.7 | partial |
| UDIS (Unsupervised Deep Image Stitching) | 2021 | 28.0 | 28.7 | done |
| APAP | 2013 | 25.0 | 28.7 | done |
| Laplacian Pyramid Fusion (PWM) | — | 16.7 | 28.7 | done |
| Guided Filter Fusion (PWM) | — | 16.7 | 28.7 | done |
| IFCNN (PWM) | — | 16.7 | 28.7 | done |
| precomputed_baseline (test) | — | 16.7 | 28.7 | done |
| Single homography stitch | 2024 | 15.5 | 28.7 | done |
