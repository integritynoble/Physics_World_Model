# Lensless (Diffuser Camera) Imaging — LenslessFormer

**GPU**  *Cao H. et al., LenslessFormer: Lensless Image Restoration via Transformer, CVPR, 2024*
**Input**: diffuser measurement (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/lensless/public/`

```python
from algorithm_base.lensless.solvers import run_solver
x = run_solver('lensless_former', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
