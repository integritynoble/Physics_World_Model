# Single-Pixel Camera (SPC) — CoSaMP

**CPU**  *Needell & Tropp, Appl. Comput. Harmon. Anal. 2009*
**Input**: photon counts (T × H × W, uint16)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/spc/public/`

```python
from algorithm_base.spc.solvers import run_solver
x = run_solver('cosamp', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
