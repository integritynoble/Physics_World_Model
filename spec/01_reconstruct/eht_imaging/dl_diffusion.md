# Event Horizon Telescope (EHT) Imaging — RS-Diffusion

**GPU**  *Diffusion model for RS, 2024*
**Input**: hologram (H × W, complex64)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/eht_imaging/public/`

```python
from algorithm_base.eht_imaging.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
