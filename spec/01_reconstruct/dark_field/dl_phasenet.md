# Dark-Field Microscopy — PhaseNet

**GPU**  *DL phase retrieval, 2018*
**Input**: grating images (2 × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/dark_field/public/`

```python
from algorithm_base.dark_field.solvers import run_solver
x = run_solver('dl_phasenet', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
