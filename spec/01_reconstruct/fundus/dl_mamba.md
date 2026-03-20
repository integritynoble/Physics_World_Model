# Fundus Camera — MedMamba

**GPU**  *SSM for medical imaging, 2026*
**Input**: photograph (H × W × 3, uint8)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/fundus/public/`

```python
from algorithm_base.fundus.solvers import run_solver
x = run_solver('dl_mamba', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
