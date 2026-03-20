# Ultrasonic Phased Array (TFM/FMC) — DiffusionMed

**GPU**  *Diffusion model for medical imaging, 2024*
**Input**: FMC data (elem × elem × time, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ultrasonic_phased_array/public/`

```python
from algorithm_base.ultrasonic_phased_array.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
