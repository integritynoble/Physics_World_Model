# Stimulated Raman Scattering (SRS) Microscopy — Spec-CNN

**GPU**  *CNN for spectroscopy, 2018*
**Input**: SRS image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/srs/public/`

```python
from algorithm_base.srs.solvers import run_solver
x = run_solver('dl_cnn', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
