# OCT Angiography (OCTA) — FFT Recon (OCTA)

**CPU**
**Input**: B-scans (T × depth × A-scans, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/octa/public/`

```python
from algorithm_base.octa.solvers import run_solver
x = run_solver('traditional_cpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
