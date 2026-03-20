# Sonar Imaging — RS-Diffusion

**GPU**  *Diffusion model for RS, 2024*
**Input**: echo data (elements × samples, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/sonar/public/`

```python
from algorithm_base.sonar.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
