# X-ray Fluorescence Tomography — PnP-ADMM (NLM)

**CPU**  *Venkatakrishnan et al., GlobalSIP 2013*
**Input**: XRF sinograms (elem × angles × det, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/xrf_tomo/public/`

```python
from algorithm_base.xrf_tomo.solvers import run_solver
cfg = {'iters': 20, 'sigma': 0.05, 'rho': 0.5}
x = run_solver('pnp_admm_nlm', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
