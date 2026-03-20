# Coded Exposure / Flutter Shutter — CS-Diffusion

**GPU**  *Diffusion for CS, 2025*
**Input**: coded frames (N × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/coded_exposure/public/`

```python
from algorithm_base.coded_exposure.solvers import run_solver
x = run_solver('dl_diffusion', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
