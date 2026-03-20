# Entangled Photon Microscopy — QGI-DL [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: coincidence counts (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/entangled_photon/public/`

```python
from algorithm_base.entangled_photon.solvers import run_solver
x = run_solver('qgi_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
