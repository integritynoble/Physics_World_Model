# Passive Microwave Radiometry — PnP-FISTA (NLM)

**CPU**  *Beck & Teboulle 2009 + PnP*
**Input**: brightness T (H × W × ch, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/passive_microwave/public/`

```python
from algorithm_base.passive_microwave.solvers import run_solver
cfg = {'iters': 20, 'sigma': 0.05, 'mu': 0.5}
x = run_solver('pnp_fista_nlm', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
