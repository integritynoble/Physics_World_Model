# Gravitational Wave Detection — RS-Mamba

**GPU**  *SSM for remote sensing, 2026*
**Input**: strain (samples, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/gravitational_wave/public/`

```python
from algorithm_base.gravitational_wave.solvers import run_solver
x = run_solver('dl_mamba', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
