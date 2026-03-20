# Scanning Electron Microscopy (SEM) — Chambolle-Pock

**CPU**  *Chambolle & Pock, JMIV 2011*
**Input**: SEM image (H × W, uint16)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/sem/public/`

```python
from algorithm_base.sem.solvers import run_solver
cfg = {'iters': 30, 'lam': 0.005}
x = run_solver('chambolle_pock', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
