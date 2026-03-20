# Single-Pixel Camera (SPC) — GPSR

**CPU**  *Figueiredo, Nowak & Wright, IEEE JSTSP 2007*
**Input**: photon counts (T × H × W, uint16)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/spc/public/`

```python
from algorithm_base.spc.solvers import run_solver
x = run_solver('gpsr', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
