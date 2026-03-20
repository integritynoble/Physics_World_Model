# Atomic Force Microscopy (AFM) — CARE

**GPU**  *Weigert et al. 2018*
**Input**: force-distance map (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/afm/public/`

```python
from algorithm_base.afm.solvers import run_solver
x = run_solver('best_quality', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
