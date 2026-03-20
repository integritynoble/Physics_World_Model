# Photometric Stereo — PS-FCN [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: images under N lights (N × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/photometric_stereo/public/`

```python
from algorithm_base.photometric_stereo.solvers import run_solver
x = run_solver('ps_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
