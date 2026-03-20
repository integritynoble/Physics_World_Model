# OCT Angiography (OCTA) — DiffusionMed

**GPU**  *Diffusion model for medical imaging, 2024*
**Input**: B-scans (T × depth × A-scans, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/octa/public/`

```python
from algorithm_base.octa.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
