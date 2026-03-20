# X-ray Computed Tomography (CT) — PnP-HQS (NLM)

**CPU**  **PSNR**: ~39.1 dB  *Zhang et al., TIP 2017 — 39.1 dB on LoDoPaB*
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/`

```python
from algorithm_base.ct.solvers import run_solver
cfg = {'iters': 15, 'sigma': 0.05}
x = run_solver('pnp_hqs_nlm', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
