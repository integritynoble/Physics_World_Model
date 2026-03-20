# Ultrasound B-mode Imaging — Delay-Multiply-and-Sum

**CPU**  *Matrone et al. 2015, IEEE TUFFC*
**Input**: RF data (elements × samples, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ultrasound/public/`

```python
from algorithm_base.ultrasound.solvers import run_solver
x = run_solver('dmas', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
