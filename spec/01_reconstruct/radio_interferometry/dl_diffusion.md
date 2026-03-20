# Radio Interferometry (VLBI) — RS-Diffusion

**GPU**  *Diffusion model for RS, 2024*
**Input**: UV-plane data (N_baselines, complex64)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/radio_interferometry/public/`

```python
from algorithm_base.radio_interferometry.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
