# Scanning Tunneling Microscopy (STM) — Probe-Transformer

**GPU**  *Transformer for probe imaging, 2023*
**Input**: tunneling map (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/stm/public/`

```python
from algorithm_base.stm.solvers import run_solver
x = run_solver('dl_transformer', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
