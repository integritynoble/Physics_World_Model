# Shearography — Probe-Transformer

**GPU**  *Transformer for probe imaging, 2023*
**Input**: shearograms (2 × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/shearography/public/`

```python
from algorithm_base.shearography.solvers import run_solver
x = run_solver('dl_transformer', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
