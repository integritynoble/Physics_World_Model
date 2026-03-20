# Small-Angle X-ray Scattering (SAXS) — SAXS-VAE [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: scattering pattern (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/saxs/public/`

```python
from algorithm_base.saxs.solvers import run_solver
x = run_solver('saxs_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
