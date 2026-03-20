# Weather / Doppler Radar — RS-Transformer

**GPU**  *Transformer for remote sensing, 2022*
**Input**: reflectivity (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/weather_radar/public/`

```python
from algorithm_base.weather_radar.solvers import run_solver
x = run_solver('dl_transformer', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
