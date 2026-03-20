# Electron Tomography — DiffusionRecon

**GPU**  *Song et al., 2024*
**Input**: tilt series (angles × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/electron_tomography/public/`

```python
from algorithm_base.electron_tomography.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
