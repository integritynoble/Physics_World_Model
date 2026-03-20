# Event Horizon Telescope (EHT) Imaging — Chambolle-Pock

**CPU**  *Chambolle & Pock, JMIV 2011*
**Input**: hologram (H × W, complex64)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/eht_imaging/public/`

```python
from algorithm_base.eht_imaging.solvers import run_solver
cfg = {'iters': 30, 'lam': 0.005}
x = run_solver('chambolle_pock', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
