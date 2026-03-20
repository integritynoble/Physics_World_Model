# Ultrasound B-mode Imaging — ABLE (PnP-HQS DRUNet)

**GPU**  *Luijten et al. 2020, Nature MI*
**Input**: RF data (elements × samples, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ultrasound/public/`

```python
from algorithm_base.ultrasound.solvers import run_solver
x = run_solver('able', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
