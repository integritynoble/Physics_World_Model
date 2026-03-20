# X-ray Crystallography — TV-ADMM

**CPU**  *Rudin, Osher & Fatemi 1992; Boyd et al. 2010*
**Input**: structure factors (hkl × F, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/xray_crystallography/public/`

```python
from algorithm_base.xray_crystallography.solvers import run_solver
cfg = {'iters': 20, 'lam': 0.005, 'rho': 1.0}
x = run_solver('tv_admm', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
