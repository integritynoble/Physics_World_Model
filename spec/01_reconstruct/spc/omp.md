# Single-Pixel Camera (SPC) — OMP

**CPU**  *Pati, Rezaiifar & Krishnaprasad, Asilomar 1993*
**Input**: photon counts (T × H × W, uint16)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/spc/public/`

```python
from algorithm_base.spc.solvers import run_solver
x = run_solver('omp', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
