# Interferometric SAR (InSAR) — InSAR-Net [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: interferometric phase (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/insar/public/`

```python
from algorithm_base.insar.solvers import run_solver
x = run_solver('insar_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
