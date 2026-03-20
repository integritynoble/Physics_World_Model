# X-ray Angiography — MedMamba

**GPU**  *SSM for medical imaging, 2026*
**Input**: projection (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/angiography/public/`

```python
from algorithm_base.angiography.solvers import run_solver
x = run_solver('dl_mamba', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
