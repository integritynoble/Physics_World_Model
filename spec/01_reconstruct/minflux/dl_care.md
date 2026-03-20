# MINFLUX Nanoscopy — CARE

**GPU**  *Weigert et al., Nat Methods 2018*
**Input**: photon records (N × 5: t,x,y,z,id)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/minflux/public/`

```python
from algorithm_base.minflux.solvers import run_solver
x = run_solver('dl_care', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
