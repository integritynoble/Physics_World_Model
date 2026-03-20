# Neural Radiance Fields (NeRF) — Mip-NeRF 360

**GPU**  *Barron et al. CVPR 2022*
**Input**: posed images (N × H × W × 3, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/nerf/public/`

```python
from algorithm_base.nerf.solvers import run_solver
x = run_solver('best_quality', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
