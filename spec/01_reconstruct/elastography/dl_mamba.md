# Shear-Wave Elastography — MedMamba

**GPU**  *SSM for medical imaging, 2026*
**Input**: displacement (H × W × 3, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/elastography/public/`

```python
from algorithm_base.elastography.solvers import run_solver
x = run_solver('dl_mamba', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
