# Dark-Field Microscopy — Phase-Diffusion

**GPU**  *Diffusion for phase retrieval, 2025*
**Input**: grating images (2 × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/dark_field/public/`

```python
from algorithm_base.dark_field.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
