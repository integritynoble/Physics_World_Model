# Polarization Microscopy — CS-Diffusion

**GPU**  *Diffusion for CS, 2025*
**Input**: Stokes images (4 × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/polarization/public/`

```python
from algorithm_base.polarization.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
