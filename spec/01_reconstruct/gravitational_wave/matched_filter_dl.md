# Gravitational Wave Detection — GW-DL (PyCBC-ML) [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: strain (samples, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/gravitational_wave/public/`

```python
from algorithm_base.gravitational_wave.solvers import run_solver
x = run_solver('matched_filter_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
