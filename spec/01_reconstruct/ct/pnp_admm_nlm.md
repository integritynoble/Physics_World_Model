# X-ray Computed Tomography (CT) — PnP-ADMM (NLM)

**CPU**  **PSNR**: ~39.5 dB  *Venkatakrishnan et al., GlobalSIP 2013 — 39.5 dB on LoDoPaB*
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/`

```python
from algorithm_base.ct.solvers import run_solver
cfg = {'iters': 20, 'sigma': 0.05, 'rho': 0.5}
x = run_solver('pnp_admm_nlm', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
