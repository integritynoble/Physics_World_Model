# Widefield Fluorescence Microscopy — Wiener Filter

**CPU**  *Wiener 1949, Extrapolation, Interpolation, and Smoothing*
**Input**: fluorescence image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/widefield/public/`

```python
from algorithm_base.widefield.solvers import run_solver
x = run_solver('wiener', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
