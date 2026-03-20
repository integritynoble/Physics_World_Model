# Correlative Light-Electron Microscopy (CLEM) — CARE

**GPU**  *Weigert et al., Nat Methods 2018*
**Input**: EM + fluorescence (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/clem/public/`

```python
from algorithm_base.clem.solvers import run_solver
x = run_solver('dl_care', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
