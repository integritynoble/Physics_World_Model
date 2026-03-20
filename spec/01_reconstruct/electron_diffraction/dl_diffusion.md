# 4D-STEM Electron Diffraction — Phase-Diffusion

**GPU**  *Diffusion for phase retrieval, 2025*
**Input**: diffraction pattern (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/electron_diffraction/public/`

```python
from algorithm_base.electron_diffraction.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
