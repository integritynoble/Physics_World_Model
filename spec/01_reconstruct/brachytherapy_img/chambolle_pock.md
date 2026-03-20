# Brachytherapy Imaging — Chambolle-Pock

**CPU**  *Chambolle & Pock, JMIV 2011*
**Input**: dose map (H × W × D, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/brachytherapy_img/public/`

```python
from algorithm_base.brachytherapy_img.solvers import run_solver
cfg = {'iters': 30, 'lam': 0.005}
x = run_solver('chambolle_pock', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
