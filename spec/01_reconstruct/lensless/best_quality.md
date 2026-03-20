# Lensless (Diffuser Camera) Imaging — FlatNet

**GPU**  *Khan S.S. et al., FlatNet: Towards Photorealistic Scene Reconstruction from Lensless Measurements, IEEE TPAMI, 2020*
**Input**: diffuser measurement (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/lensless/public/`

```python
from algorithm_base.lensless.solvers import run_solver
x = run_solver('best_quality', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
