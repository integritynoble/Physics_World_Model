# Raman Imaging / Microscopy — Spec-CNN

**GPU**  *CNN for spectroscopy, 2018*
**Input**: Raman spectra (H × W × wn, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/raman_imaging/public/`

```python
from algorithm_base.raman_imaging.solvers import run_solver
x = run_solver('dl_cnn', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
