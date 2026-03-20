# Integral Photography — CS-Transformer

**GPU**  *Transformer for CS, 2023*
**Input**: integral image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/integral/public/`

```python
from algorithm_base.integral.solvers import run_solver
x = run_solver('dl_transformer', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
