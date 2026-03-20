# High Dynamic Range (HDR) Imaging — Chambolle-Pock

**CPU**  *Chambolle & Pock, JMIV 2011*
**Input**: multi-exposure (K × H × W × 3, uint8)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/hdr_imaging/public/`

```python
from algorithm_base.hdr_imaging.solvers import run_solver
cfg = {'iters': 30, 'lam': 0.005}
x = run_solver('chambolle_pock', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
