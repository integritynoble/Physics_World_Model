# Mammography — MedMamba

**GPU**  *SSM for medical imaging, 2026*
**Input**: projection pair (2 × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mammography/public/`

```python
from algorithm_base.mammography.solvers import run_solver
x = run_solver('dl_mamba', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
