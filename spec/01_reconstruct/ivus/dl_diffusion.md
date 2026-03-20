# Intravascular Ultrasound (IVUS) — DiffusionMed

**GPU**  *Diffusion model for medical imaging, 2024*
**Input**: RF pullback (frames × elements × samples, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ivus/public/`

```python
from algorithm_base.ivus.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
