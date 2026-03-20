# Wide-Angle X-ray Scattering (WAXS) — Phase-Diffusion

**GPU**  *Diffusion for phase retrieval, 2025*
**Input**: wide-angle pattern (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/waxs/public/`

```python
from algorithm_base.waxs.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
