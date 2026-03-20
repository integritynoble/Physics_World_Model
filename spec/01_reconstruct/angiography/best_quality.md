# X-ray Angiography — DSA-Net [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: projection (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/angiography/public/`

```python
from algorithm_base.angiography.solvers import run_solver
x = run_solver('best_quality', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
