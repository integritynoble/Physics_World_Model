# Diffuse Optical Tomography (DOT) — MambaRecon

**GPU**  *SSM for inverse problems, 2026*
**Input**: boundary flux (sources × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/dot/public/`

```python
from algorithm_base.dot.solvers import run_solver
x = run_solver('dl_mamba', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
