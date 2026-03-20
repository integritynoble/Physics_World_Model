# Three-Photon Microscopy — Chambolle-Pock

**CPU**  *Chambolle & Pock, JMIV 2011*
**Input**: Z-stack (Z × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/three_photon/public/`

```python
from algorithm_base.three_photon.solvers import run_solver
cfg = {'iters': 30, 'lam': 0.005}
x = run_solver('chambolle_pock', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
