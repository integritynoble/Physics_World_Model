# Fluoroscopy — DiffusionMed

**GPU**  *Diffusion model for medical imaging, 2024*
**Input**: X-ray frames (T × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/fluoroscopy/public/`

```python
from algorithm_base.fluoroscopy.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
