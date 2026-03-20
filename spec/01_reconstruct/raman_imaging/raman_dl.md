# Raman Imaging / Microscopy — RamanNet [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: Raman spectra (H × W × wn, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/raman_imaging/public/`

```python
from algorithm_base.raman_imaging.solvers import run_solver
x = run_solver('raman_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
