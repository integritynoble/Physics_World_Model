# Three-Photon Microscopy — DiffusionMicro

**GPU**  *Diffusion-based microscopy, 2025*
**Input**: Z-stack (Z × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/three_photon/public/`

```python
from algorithm_base.three_photon.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
