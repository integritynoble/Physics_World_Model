# Fiber Bundle Endoscopy — AF-SfMLearner

**GPU**  *Shao, S. et al. (2022) Self-supervised depth estimation in endoscopy, MICCAI 2022*
**Input**: image (H × W × 3, uint8)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/endoscopy/public/`

```python
from algorithm_base.endoscopy.solvers import run_solver
x = run_solver('famous_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
