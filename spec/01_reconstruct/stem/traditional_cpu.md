# Scanning Transmission Electron Microscopy (STEM) — Richardson-Lucy (STEM)

**CPU**
**Input**: HAADF image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/stem/public/`

```python
from algorithm_base.stem.solvers import run_solver
x = run_solver('traditional_cpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
