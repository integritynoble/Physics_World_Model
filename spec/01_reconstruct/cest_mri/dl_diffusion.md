# CEST MRI — DiffusionMed

**GPU**  *Diffusion model for medical imaging, 2024*
**Input**: Z-spectrum (offsets × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cest_mri/public/`

```python
from algorithm_base.cest_mri.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
