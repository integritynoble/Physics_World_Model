# Light Field Imaging — CS-Diffusion

**GPU**  *Diffusion for CS, 2025*
**Input**: light field (u × v × s × t, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/light_field/public/`

```python
from algorithm_base.light_field.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
