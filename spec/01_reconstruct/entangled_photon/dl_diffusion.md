# Entangled Photon Microscopy — CS-Diffusion

**GPU**  *Diffusion for CS, 2025*
**Input**: coincidence counts (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/entangled_photon/public/`

```python
from algorithm_base.entangled_photon.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
