# Bioluminescence Tomography (BLT) — PnP-ADMM (NLM)

**CPU**  *Venkatakrishnan et al., GlobalSIP 2013*
**Input**: surface flux (H × W × angles, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/bioluminescence_tomo/public/`

```python
from algorithm_base.bioluminescence_tomo.solvers import run_solver
cfg = {'iters': 20, 'sigma': 0.05, 'rho': 0.5}
x = run_solver('pnp_admm_nlm', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
