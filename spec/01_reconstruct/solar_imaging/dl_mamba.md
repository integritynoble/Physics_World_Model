# Solar EUV/X-ray Imaging — RS-Mamba

**GPU**  *SSM for remote sensing, 2026*
**Input**: EUV image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/solar_imaging/public/`

```python
from algorithm_base.solar_imaging.solvers import run_solver
x = run_solver('dl_mamba', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
