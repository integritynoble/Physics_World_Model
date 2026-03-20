# Polarimetric SAR (PolSAR) — PolSAR-Net [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: scattering matrix (H × W × 4, complex64)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/polsar/public/`

```python
from algorithm_base.polsar.solvers import run_solver
x = run_solver('polsar_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
