# Diffusion MRI (DTI) — Landweber Iteration

**CPU**  *Landweber, Am J Math 1951*
**Input**: DWI (N_dirs × H × W × D, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/diffusion_mri/public/`

```python
from algorithm_base.diffusion_mri.solvers import run_solver
cfg = {'iters': 50, 'step': 0.5}
x = run_solver('landweber', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
