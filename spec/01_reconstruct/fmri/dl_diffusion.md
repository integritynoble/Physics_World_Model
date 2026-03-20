# Functional MRI (BOLD fMRI) — DiffusionMed

**GPU**  *Diffusion model for medical imaging, 2024*
**Input**: BOLD volumes (T × H × W × D, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/fmri/public/`

```python
from algorithm_base.fmri.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
