# Talbot-Lau X-ray Grating Interferometry — Phase-Diffusion

**GPU**  *Diffusion for phase retrieval, 2025*
**Input**: stepping images (N_steps × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/talbot_lau/public/`

```python
from algorithm_base.talbot_lau.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
