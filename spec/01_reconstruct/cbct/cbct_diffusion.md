# Cone-Beam Computed Tomography (CBCT) — CBCT Diffusion (DL-PGD)

**GPU**  *Chung, H. et al. (2023) Solving 3D inverse problems using pre-trained 2D diffusion models, CVPR*
**Input**: projections (angles × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cbct/public/`

```python
from algorithm_base.cbct.solvers import run_solver
x = run_solver('cbct_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
