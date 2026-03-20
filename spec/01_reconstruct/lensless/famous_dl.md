# Lensless (Diffuser Camera) Imaging — Le-ADMM-U

**GPU**  *Monakhova K. et al., Learned Reconstructions for Practical Mask-Based Lensless Imaging, IEEE TPAMI, 2022*
**Input**: diffuser measurement (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/lensless/public/`

```python
from algorithm_base.lensless.solvers import run_solver
x = run_solver('famous_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
