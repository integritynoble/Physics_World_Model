# Single-Pixel Camera (SPC) — FISTA-L1

**CPU**  *Beck & Teboulle, SIAM J. Imaging Sci. 2009*
**Input**: photon counts (T × H × W, uint16)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/spc/public/`

```python
from algorithm_base.spc.solvers import run_solver
x = run_solver('fista_l1', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
