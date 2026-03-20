# Proton Therapy Imaging — FBP [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: dose distribution (H × W × D, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/proton_therapy_img/public/`

```python
from algorithm_base.proton_therapy_img.solvers import run_solver
x = run_solver('traditional_cpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
