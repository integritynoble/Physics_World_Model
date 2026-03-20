# Structured Illumination Microscopy (SIM) — PnP-ADMM (NLM)

**CPU**  *Venkatakrishnan et al., GlobalSIP 2013*
**Input**: raw frames (9 × H × W: 3 angles × 3 phases)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/sim/public/`

```python
from algorithm_base.sim.solvers import run_solver
cfg = {'iters': 20, 'sigma': 0.05, 'rho': 0.5}
x = run_solver('pnp_admm_nlm', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
