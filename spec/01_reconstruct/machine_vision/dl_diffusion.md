# Machine Vision / AOI — DL-Diffusion

**GPU**  *Diffusion reconstruction, 2025*
**Input**: image (H × W × 3, uint8)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/machine_vision/public/`

```python
from algorithm_base.machine_vision.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
