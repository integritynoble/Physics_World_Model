# Optical Coherence Tomography (OCT) — OCT Denoising Net

**CPU**
**Input**: spectrum (wavenumbers × A-scans, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/oct/public/`

```python
from algorithm_base.oct.solvers import run_solver
x = run_solver('small_gpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
