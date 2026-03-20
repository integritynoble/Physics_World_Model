# Transmission Electron Microscopy (TEM) — 3D-Transformer

**GPU**  *Transformer for 3D, 2023*
**Input**: TEM image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/tem/public/`

```python
from algorithm_base.tem.solvers import run_solver
x = run_solver('dl_transformer', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
