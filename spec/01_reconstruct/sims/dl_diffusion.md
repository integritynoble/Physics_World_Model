# Secondary Ion Mass Spectrometry (SIMS) Imaging — Spec-Diffusion

**GPU**  *Diffusion for spectroscopy, 2025*
**Input**: ion images (H × W × m/z, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/sims/public/`

```python
from algorithm_base.sims.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
