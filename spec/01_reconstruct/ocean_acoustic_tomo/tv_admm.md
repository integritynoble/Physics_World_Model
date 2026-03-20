# Ocean Acoustic Tomography — TV-ADMM

**CPU**  *Rudin, Osher & Fatemi 1992; Boyd et al. 2010*
**Input**: travel times (pairs, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ocean_acoustic_tomo/public/`

```python
from algorithm_base.ocean_acoustic_tomo.solvers import run_solver
cfg = {'iters': 20, 'lam': 0.005, 'rho': 1.0}
x = run_solver('tv_admm', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
