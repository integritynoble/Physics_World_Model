# Synthetic Aperture Radar (SAR) — SAR-CNN [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: raw data (range × azimuth, complex64)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/sar/public/`

```python
from algorithm_base.sar.solvers import run_solver
x = run_solver('famous_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
