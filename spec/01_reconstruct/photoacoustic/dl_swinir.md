# Photoacoustic Imaging — SwinIR-Med

**GPU**  *Liang et al., ICCV 2021*
**Input**: time-series (elements × time, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/photoacoustic/public/`

```python
from algorithm_base.photoacoustic.solvers import run_solver
x = run_solver('dl_swinir', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
