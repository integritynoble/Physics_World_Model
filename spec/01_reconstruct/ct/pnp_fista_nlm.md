# X-ray Computed Tomography (CT) — PnP-FISTA (NLM)

**CPU**  *Beck & Teboulle, SIIMS 2009 + PnP*
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/`

```python
from algorithm_base.ct.solvers import run_solver
cfg = {'iters': 20, 'sigma': 0.05}
x = run_solver('pnp_fista_nlm', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
