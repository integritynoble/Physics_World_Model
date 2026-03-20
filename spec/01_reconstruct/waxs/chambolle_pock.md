# Wide-Angle X-ray Scattering (WAXS) — Chambolle-Pock

**CPU**  *Chambolle & Pock, JMIV 2011*
**Input**: wide-angle pattern (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/waxs/public/`

```python
from algorithm_base.waxs.solvers import run_solver
cfg = {'iters': 30, 'lam': 0.005}
x = run_solver('chambolle_pock', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
