# Single-Pixel Camera (SPC) — Richardson-Lucy

**CPU**  *Richardson, JOSA 1972; Lucy, Astron. J. 1974*
**Input**: photon counts (T × H × W, uint16)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/spc/public/`

```python
from algorithm_base.spc.solvers import run_solver
x = run_solver('richardson_lucy', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
