# Lucky Imaging — DL-UNet

**GPU**  *U-Net reconstruction, 2018*
**Input**: speckle frames (N × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/lucky_imaging/public/`

```python
from algorithm_base.lucky_imaging.solvers import run_solver
x = run_solver('dl_unet', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
