# Hyperspectral Remote Sensing — RS-Diffusion

**GPU**  *Diffusion model for RS, 2024*
**Input**: radiance cube (H × W × bands, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/hyperspectral_remote/public/`

```python
from algorithm_base.hyperspectral_remote.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
