# Brachytherapy Imaging — BrachyNet [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: dose map (H × W × D, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/brachytherapy_img/public/`

```python
from algorithm_base.brachytherapy_img.solvers import run_solver
x = run_solver('brachy_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
