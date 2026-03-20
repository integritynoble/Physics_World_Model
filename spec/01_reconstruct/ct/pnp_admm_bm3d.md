# X-ray Computed Tomography (CT) — PnP-ADMM (BM3D)

**CPU**  *Venkatakrishnan et al. 2013 + Dabov et al. TIP 2007*
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/`

```python
from algorithm_base.ct.solvers import run_solver
cfg = {'iters': 10, 'sigma': 0.05, 'rho': 0.5}
x = run_solver('pnp_admm_bm3d', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
