# Stimulated Raman Scattering (SRS) Microscopy — Chambolle-Pock

**CPU**  *Chambolle & Pock, JMIV 2011*
**Input**: SRS image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/srs/public/`

```python
from algorithm_base.srs.solvers import run_solver
cfg = {'iters': 30, 'lam': 0.005}
x = run_solver('chambolle_pock', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
