# Lensless (Diffuser Camera) Imaging — LensMamba

**GPU**  *Mamba-based lensless imaging reconstruction with state-space modelling, 2024*
**Input**: diffuser measurement (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/lensless/public/`

```python
from algorithm_base.lensless.solvers import run_solver
x = run_solver('lens_mamba', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
