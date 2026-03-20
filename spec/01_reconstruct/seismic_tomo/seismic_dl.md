# Seismic Tomography — SeisInversion-Net [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: travel times (src-recv, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/seismic_tomo/public/`

```python
from algorithm_base.seismic_tomo.solvers import run_solver
x = run_solver('seismic_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
