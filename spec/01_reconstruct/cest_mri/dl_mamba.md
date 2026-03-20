# CEST MRI — MedMamba

**GPU**  *SSM for medical imaging, 2026*
**Input**: Z-spectrum (offsets × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cest_mri/public/`

```python
from algorithm_base.cest_mri.solvers import run_solver
x = run_solver('dl_mamba', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
