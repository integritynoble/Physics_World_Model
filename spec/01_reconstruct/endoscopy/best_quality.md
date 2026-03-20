# Fiber Bundle Endoscopy — EndoMapper-Net

**GPU**  *Ozyoruk, K.B. et al. (2021) EndoMapper, Nat. Mach. Intel. 3*
**Input**: image (H × W × 3, uint8)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/endoscopy/public/`

```python
from algorithm_base.endoscopy.solvers import run_solver
x = run_solver('best_quality', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
