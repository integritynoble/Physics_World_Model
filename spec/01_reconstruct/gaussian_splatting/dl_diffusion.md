# 3D Gaussian Splatting (3DGS) — 3D-Diffusion

**GPU**  *Diffusion for 3D reconstruction, 2025*
**Input**: posed images (N × H × W × 3, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/gaussian_splatting/public/`

```python
from algorithm_base.gaussian_splatting.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
