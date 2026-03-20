# Confocal Laser Endomicroscopy (CLE) — DiffusionMicro

**GPU**  *Diffusion-based microscopy, 2025*
**Input**: confocal frame (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/confocal_endomicroscopy/public/`

```python
from algorithm_base.confocal_endomicroscopy.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
