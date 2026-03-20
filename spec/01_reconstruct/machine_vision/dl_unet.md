# Machine Vision / AOI — DL-UNet

**GPU**  *U-Net reconstruction, 2018*
**Input**: image (H × W × 3, uint8)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/machine_vision/public/`

```python
from algorithm_base.machine_vision.solvers import run_solver
x = run_solver('dl_unet', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
