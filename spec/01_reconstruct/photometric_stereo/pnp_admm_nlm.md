# Photometric Stereo — PnP-ADMM (NLM)

**CPU**  *Venkatakrishnan et al., GlobalSIP 2013*
**Input**: images under N lights (N × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/photometric_stereo/public/`

```python
from algorithm_base.photometric_stereo.solvers import run_solver
cfg = {'iters': 20, 'sigma': 0.05, 'rho': 0.5}
x = run_solver('pnp_admm_nlm', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
