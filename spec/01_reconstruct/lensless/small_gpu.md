# Lensless (Diffuser Camera) Imaging — FlatNet-Lite

**GPU**  *Khan S.S. et al., FlatNet: Towards Photorealistic Scene Reconstruction from Lensless Measurements, IEEE TPAMI, 2020*
**Input**: diffuser measurement (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/lensless/public/`

```python
from algorithm_base.lensless.solvers import run_solver
x = run_solver('small_gpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
