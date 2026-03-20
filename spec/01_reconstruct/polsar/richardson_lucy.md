# Polarimetric SAR (PolSAR) — Richardson-Lucy

**CPU**  *Richardson 1972; Lucy 1974*
**Input**: scattering matrix (H × W × 4, complex64)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/polsar/public/`

```python
from algorithm_base.polsar.solvers import run_solver
cfg = {'iters': 50}
x = run_solver('richardson_lucy', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
