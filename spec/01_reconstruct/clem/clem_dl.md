# Correlative Light-Electron Microscopy (CLEM) — CLEM-Net [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: EM + fluorescence (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/clem/public/`

```python
from algorithm_base.clem.solvers import run_solver
x = run_solver('clem_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
