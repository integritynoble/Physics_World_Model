# Ultrasound B-mode Imaging — Wiener Filter

**CPU**  *Wiener 1949, Extrapolation, Interpolation, and Smoothing*
**Input**: RF data (elements × samples, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ultrasound/public/`

```python
from algorithm_base.ultrasound.solvers import run_solver
x = run_solver('wiener', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
