# FTIR Spectroscopic Imaging — FTIR-UNet [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: interferogram (H × W × OPD, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ftir_imaging/public/`

```python
from algorithm_base.ftir_imaging.solvers import run_solver
x = run_solver('ftir_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
