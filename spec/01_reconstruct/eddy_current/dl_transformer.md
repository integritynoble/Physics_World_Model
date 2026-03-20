# Eddy Current Imaging — Probe-Transformer

**GPU**  *Transformer for probe imaging, 2023*
**Input**: induced voltage (coils × time, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/eddy_current/public/`

```python
from algorithm_base.eddy_current.solvers import run_solver
x = run_solver('dl_transformer', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
