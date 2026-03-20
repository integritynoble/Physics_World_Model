# Phase Contrast Microscopy — Phase-Diffusion

**GPU**  *Diffusion for phase retrieval, 2025*
**Input**: image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/phase_contrast/public/`

```python
from algorithm_base.phase_contrast.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
