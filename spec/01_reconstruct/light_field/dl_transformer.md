# Light Field Imaging — CS-Transformer

**GPU**  *Transformer for CS, 2023*
**Input**: light field (u × v × s × t, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/light_field/public/`

```python
from algorithm_base.light_field.solvers import run_solver
x = run_solver('dl_transformer', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
