# Digital Holographic Microscopy — Angular Spectrum Method

**CPU**  *Goodman J.W., Introduction to Fourier Optics, McGraw-Hill, 1968 (angular spectrum propagation, 1960s)*
**Input**: hologram (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/holography/public/`

```python
from algorithm_base.holography.solvers import run_solver
x = run_solver('traditional_cpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
