# Ptychographic Imaging — TV-ADMM

**CPU**  *Boyd, S. et al. (2008/2011) Distributed optimization and statistical learning via ADMM, Foundations and Trends in ML*
**Input**: diffraction patterns (N_pos × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ptychography/public/`

```python
from algorithm_base.ptychography.solvers import run_solver
x = run_solver('tv_admm', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
