# Seismic Tomography — Chambolle-Pock

**CPU**  *Chambolle & Pock, JMIV 2011*
**Input**: travel times (src-recv, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/seismic_tomo/public/`

```python
from algorithm_base.seismic_tomo.solvers import run_solver
cfg = {'iters': 30, 'lam': 0.005}
x = run_solver('chambolle_pock', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
