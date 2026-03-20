# Radio Aperture Synthesis — Chambolle-Pock

**CPU**  *Chambolle & Pock, JMIV 2011*
**Input**: visibilities (baselines × freq × T, complex64)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/radio_astronomy/public/`

```python
from algorithm_base.radio_astronomy.solvers import run_solver
cfg = {'iters': 30, 'lam': 0.005}
x = run_solver('chambolle_pock', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
