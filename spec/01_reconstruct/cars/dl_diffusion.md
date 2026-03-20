# Coherent Anti-Stokes Raman (CARS) Microscopy — Spec-Diffusion

**GPU**  *Diffusion for spectroscopy, 2025*
**Input**: CARS spectrum cube (H × W × λ, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cars/public/`

```python
from algorithm_base.cars.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
