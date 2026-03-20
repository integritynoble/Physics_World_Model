# Lensless (Diffuser Camera) Imaging — PhlatCam

**GPU**  *Boominathan V. et al., PhlatCam: Designed Phase-Mask Based Thin Lensless Camera, IEEE TPAMI / ICCP, 2020*
**Input**: diffuser measurement (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/lensless/public/`

```python
from algorithm_base.lensless.solvers import run_solver
x = run_solver('phlatcam', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
