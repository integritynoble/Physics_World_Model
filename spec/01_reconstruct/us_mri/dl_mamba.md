# US/MRI Fusion — MedMamba

**GPU**  *SSM for medical imaging, 2026*
**Input**: US + MRI combined data
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/us_mri/public/`

```python
from algorithm_base.us_mri.solvers import run_solver
x = run_solver('dl_mamba', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
