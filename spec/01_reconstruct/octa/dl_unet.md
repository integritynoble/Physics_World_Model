# OCT Angiography (OCTA) — Med-UNet

**GPU**  *Ronneberger et al., MICCAI 2015*
**Input**: B-scans (T × depth × A-scans, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/octa/public/`

```python
from algorithm_base.octa.solvers import run_solver
x = run_solver('dl_unet', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
