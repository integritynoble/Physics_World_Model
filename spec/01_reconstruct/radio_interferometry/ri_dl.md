# Radio Interferometry (VLBI) — R2D2 (interferometry) [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: UV-plane data (N_baselines, complex64)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/radio_interferometry/public/`

```python
from algorithm_base.radio_interferometry.solvers import run_solver
x = run_solver('ri_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
