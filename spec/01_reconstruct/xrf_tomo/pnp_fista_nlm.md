# X-ray Fluorescence Tomography — PnP-FISTA (NLM)

**CPU**  *Beck & Teboulle 2009 + PnP*
**Input**: XRF sinograms (elem × angles × det, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/xrf_tomo/public/`

```python
from algorithm_base.xrf_tomo.solvers import run_solver
cfg = {'iters': 20, 'sigma': 0.05, 'mu': 0.5}
x = run_solver('pnp_fista_nlm', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
