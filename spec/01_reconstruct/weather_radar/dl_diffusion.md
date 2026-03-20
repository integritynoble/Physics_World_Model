# Weather / Doppler Radar — RS-Diffusion

**GPU**  *Diffusion model for RS, 2024*
**Input**: reflectivity (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/weather_radar/public/`

```python
from algorithm_base.weather_radar.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
