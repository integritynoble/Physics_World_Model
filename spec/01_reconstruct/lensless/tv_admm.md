# Lensless (Diffuser Camera) Imaging — TV-ADMM Deconvolution

**CPU**  *Boyd S. et al., Distributed Optimization and Statistical Learning via ADMM, Foundations and Trends in ML, 2011; Chambolle A., An algorithm for TV minimization, JMIV, 2004*
**Input**: diffuser measurement (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/lensless/public/`

```python
from algorithm_base.lensless.solvers import run_solver
x = run_solver('tv_admm', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
