# Polarization Microscopy — PnP-HQS

**CPU**
**Input**: Stokes images (4 × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/polarization/public/`

```python
from algorithm_base.polarization.solvers import run_solver
x = run_solver('traditional_cpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
